import torch
import esm
from transformers import EsmTokenizer, EsmForMaskedLM
from dataclasses import dataclass,field
import os, inspect, subprocess
from Bio.PDB import PDBParser,PDBIO, Structure, PDBIO, Select,NeighborSearch, PDBList
import numpy as np
from typing import List
import json
import tyro
import warnings
from models.ours.Model import MyModel
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB import MMCIFParser

warnings.filterwarnings("ignore")

BLAST = './lib/psiblast'
BLAST_DB = '/home2/public/database/uniref90/uniref90'
HHBLITS = './lib/hhblits'
HH_DB = '/home2/public/database/UniRef30_2023_02/UniRef30_2023_02'
DSSP='./lib/dssp'
FOLDSEEK='./lib/foldseek'  

@dataclass
class Args:
    task_id:str="Undefined"
    task_name:str="Undefined"
    input_dir:str="Undefined"
    output_dir:str="Undefined"
    prot_id:str="Undefined"
    chain_id:str="Undefined"
    annotate_threshold:float=0
    map_d_threshold:float=0
    """The distance threshold for distance map"""
    map_k_number:int=0
    """The number of nearest neighbors for KNN map"""
    split_start:int=0
    """The start residue index of the fregment"""
    split_end:int=0
    """The end residue index of the fregment"""
    prot_seq:str="Undefined"
    ligand_typ:str="Undefined"
    feat_typ:List[str]=field(default_factory=list)

#未来：修改，一个文件错误整个task失败，的问题
class CustomError(Exception):
    pass

class ChainNotFoundError(CustomError):
    def __init__(self, filename, chains, request_chain):
        self.filename = filename
        self.chains = chains
        self.request_chain = request_chain
        self.message = f"Chain {request_chain} not found in {filename}. The valid chains are {chains}"
        super().__init__(self.message)

class ResIndexOutOfRangeError(CustomError):
    def __init__(self, filename, chain_id, range, request_range):
        self.filename = filename
        self.chain_id = chain_id
        self.range = range
        self.request_range = request_range
        self.message = f"Residue index range {request_range} out of range for chain {chain_id} in {filename}. The valid range is {range}"
        super().__init__(self.message)

class NotPNAComplexError(CustomError):
    def __init__(self, complex_name):
        self.prot_name = complex_name
        self.message = f"{complex_name} is not a protein-nucleic acid complex."
        super().__init__(self.message)

class InvalidStructureError(CustomError):
    def __init__(self, filename):
        self.filename = filename
        self.message = f"{filename} is not a valid PDB or CIF file."
        super().__init__(self.message)

class ResidueRangeSelect(Select):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def accept_residue(self, residue):
        res_id = residue.id[1]
        return residue.id[0] == ' ' and self.start <= res_id <= self.end

class FeatExtract:
    
    def run_esm2(args):

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device=torch.device('cpu')
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        model = model.to(device)
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        data=[]
        inputs=os.listdir(args.input_dir)
        for fasta_file in inputs:
            with open(os.path.join(args.input_dir,fasta_file),'r') as f:
                for line in f:
                    if not line.startswith('>'):
                        seq=line.strip()
                        data.append((fasta_file,seq))


        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens=batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[30], return_contacts=True)
                
        token_representations = results["representations"][30]
        prot_rep=token_representations[:,1:-1,:]
        for tuple, rep in zip(data,prot_rep):
            filename,ext=os.path.splitext(tuple[0])
            # prot_length=len(tuple[1])
            # assert prot_length==rep.shape[0]
            tmp_path=os.path.join(args.output_dir,f'{filename}_tmp.esm2')
            save_path=os.path.join(args.output_dir,f'{filename}.esm2')
            torch.save(rep, tmp_path)
            os.rename(tmp_path,save_path)
    
    def run_saprot(args):
        
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device=torch.device('cpu')
        model_path = "./models/SaProt_35M_AF2"
        tokenizer = EsmTokenizer.from_pretrained(model_path)
        model = EsmForMaskedLM.from_pretrained(model_path)
        model.to(device)

        # Get structural seqs from pdb file
        def get_struc_seq(foldseek,
                        path,
                        chains: list = None,
                        process_id: int = 0,
                        plddt_path: str = None,
                        plddt_threshold: float = 70.) -> dict:
            """
            
            Args:
                foldseek: Binary executable file of foldseek
                path: Path to pdb file
                chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
                process_id: Process ID for temporary files. This is used for parallel processing.
                plddt_path: Path to plddt file. If None, plddt will not be used.
                plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

            Returns:
                seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
                (seq, struc_seq, combined_seq).
            """
            assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
            assert os.path.exists(path), f"Pdb file not found: {path}"
            assert plddt_path is None or os.path.exists(plddt_path), f"Plddt file not found: {plddt_path}"
            prot_name=os.path.basename(path).split('.')[0]

            tmp_save_path = f"get_struc_seq_{prot_name}.tsv"
            cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
            os.system(cmd)

            seq_dict = {}
            name = os.path.basename(path)
            with open(tmp_save_path, "r") as r:
                for i, line in enumerate(r):
                    desc, seq, struc_seq = line.split("\t")[:3]
                    
                    # Mask low plddt
                    if plddt_path is not None:
                        with open(plddt_path, "r") as r:
                            plddts = np.array(json.load(r)["confidenceScore"])
                            
                            # Mask regions with plddt < threshold
                            indices = np.where(plddts < plddt_threshold)[0]
                            np_seq = np.array(list(struc_seq))
                            np_seq[indices] = "#"
                            struc_seq = "".join(np_seq)
                    
                    name_chain = desc.split(" ")[0]
                    chain = name_chain.replace(name, "").split("_")[-1]

                    if chains is None or chain in chains:
                        if chain not in seq_dict:
                            combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                            seq_dict[chain] = (seq, struc_seq, combined_seq)
                
            os.remove(tmp_save_path)
            os.remove(tmp_save_path + ".dbtype")
            return seq_dict

        inputs=os.listdir(args.input_dir)
        for pdb_file in inputs:
            filename,ext=os.path.splitext(pdb_file)
            if '_' in filename: 
                filename=filename.split('_')[0] 

            pdb_path=os.path.join(args.input_dir,pdb_file)
            # pLDDT is used to mask low-confidence regions if "plddt_path" is provided 
            seq_dict=get_struc_seq("./lib/foldseek", pdb_path)
            for chain_id, parsed_seqs in seq_dict.items():
                seq, foldseek_seq, combined_seq = parsed_seqs
                
                tokens = tokenizer.tokenize(combined_seq)

                inputs = tokenizer(combined_seq, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                outputs = outputs.logits.squeeze(0)
                outputs = outputs[1:-1,:]

                tmp_path=os.path.join(args.output_dir,f'{filename}_{chain_id}_tmp.saprot')
                save_path=os.path.join(args.output_dir,f'{filename}_{chain_id}.saprot')
                torch.save(outputs,tmp_path)
                os.rename(tmp_path,save_path)
    
    def get_handcraft_features(args):

        inputs=os.listdir(args.input_dir)
        for pdb_file in inputs:
            pdb_path=os.path.join(args.input_dir,pdb_file)
            filename,ext=os.path.splitext(pdb_file)
            seq=FeatExtract._get_seq(pdb_path,args.output_dir)
            if 'dssp' in args.feat_typ:
                FeatExtract._run_dssp(pdb_path,filename,args.output_dir)

    def _run_BLAST(fasta_file,pid,output_dir):
        outfmt_type = 5
        num_iter = 3
        evalue_threshold = 0.001
        xml_file = f"{output_dir}/{pid}.xml"
        pssm_file = f"{output_dir}/{pid}.pssm"
        if os.path.isfile(pssm_file):
            pass
        else:
            cmd = [BLAST,
                    '-query', fasta_file,
                    '-db',BLAST_DB,
                    '-out',xml_file,
                    '-evalue',str(evalue_threshold),
                    '-num_iterations',str(num_iter),
                    '-outfmt',str(outfmt_type),
                    '-out_ascii_pssm',pssm_file,  # Write the pssm file
                    '-num_threads','4']                       
            subprocess.run(cmd)

    def _run_HHblits(fasta_file, pid,output_dir):
        hhm_file = f"{output_dir}/{pid}.hhm"#ohhm
        if os.path.isfile(hhm_file):
            pass
        else:
            cmd=[HHBLITS,
                '-i',fasta_file,
                '-d',HH_DB,
                '-ohhm',hhm_file,
                '-cpu','4',
                '-v','0']
            subprocess.run(cmd)

    def _run_dssp(chain_file,pid,output_dir):
        dssp_file=f'{output_dir}/{pid}.dssp'
        if os.path.isfile(dssp_file):
            pass
        else:
            cmd=[DSSP,
                 '-i',chain_file,
                 -'o',dssp_file]
            subprocess.run(cmd)

    def _run_msms():
        pass

    def _get_pssm(pssm_file):
        with open(pssm_file,'r') as f:
            text = f.readlines()
        pssm = []
        for line in text[3:]:
            if line=='\n':
                break
            else:
                res_pssm = np.array(list(map(int,line.split()[2:22]))).reshape(1,-1)
                pssm.append(res_pssm)
        pssm = np.concatenate(pssm,axis=0)
        pssm = 1/(1+np.exp(-pssm))

        return pssm

    def _get_hmm(hmm_file):
        with open(hmm_file,'r') as f:
            text = f.readlines()
        hmm_begin_line = 0
        hmm_end_line = 0
        for i in range(len(text)):
            if '#' in text[i]:
                hmm_begin_line = i + 5
            elif '//' in text[i]:
                hmm_end_line = i
        hmm = np.zeros([int((hmm_end_line - hmm_begin_line) / 3), 30])

        axis_x = 0
        for i in range(hmm_begin_line, hmm_end_line, 3):
            line1 = text[i].split()[2:-1]
            line2 = text[i + 1].split()
            axis_y = 0
            for j in line1:
                if j == '*':
                    hmm[axis_x][axis_y] = 9999 / 10000.0
                else:
                    hmm[axis_x][axis_y] = float(j) / 10000.0
                axis_y += 1
            for j in line2:
                if j == '*':
                    hmm[axis_x][axis_y] = 9999 / 10000.0
                else:
                    hmm[axis_x][axis_y] = float(j) / 10000.0
                axis_y += 1
            axis_x += 1
        hmm = (hmm - np.min(hmm)) / (np.max(hmm) - np.min(hmm))

        return hmm

    def _get_resdepth(pdb_file,pid,output_dir):
        save_path=f"{output_dir}/{pid}.resdepth"

        _, ext=os.path.splitext(pdb_file)
        if ext==".cif":
            parser = MMCIFParser()
        else:
            parser = PDBParser()

        structure = parser.get_structure(pid, pdb_file)
        model=structure[0]
        rd=ResidueDepth(model)

        with open(save_path,'w') as f:
            for item in rd.property_list:
                f.write(f"{item[0].get_id()[1]} {item[1][0]} {item[1][1]}\n")

        return rd.property_list
    
    def _get_seq(pdb_file,output_dir):
        AA_dic = {'GLY':'G','ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','TYR':'Y','ASP':'D','ASN':'N',
        'GLU':'E','LYS':'K','GLN':'Q','MET':'M','SER':'S','THR':'T','CYS':'C','PRO':'P','HIS':'H','ARG':'R'}
        filename,ext=os.path.splitext(os.path.basename(pdb_file))
        if ext==".cif":
            parser = MMCIFParser()
        else:
            parser = PDBParser()

        structure = parser.get_structure("tmp", pdb_file)
        model=structure[0]
        seq=""
        for residue in model.get_residues():
            if residue.id[0]==' ':
                seq+=AA_dic[residue.resname]
        save_path=os.path.join(output_dir,f'{filename}.fasta')
        with open(save_path,'w') as f:
            f.write(f'>{filename}\n{seq}\n')
        
        return seq
    
    def _get_edges(pdb_file):
        prot_name=os.path.basename(pdb_file).split('.')[0]
        chain_id=prot_name.split('_')[1]

        _, ext=os.path.splitext(pdb_file)
        if ext==".cif":
            parser = MMCIFParser()
        else:
            parser = PDBParser()

        structure=parser.get_structure(prot_name,pdb_file)
        edges=[]
        edge_attr=[]
        left=[] 
        right=[]
        coords=[]
        for residue in structure[0][chain_id]:
            if residue.id[0]==' ': #标准残基
                atom_coords=[atom.coord for atom in residue]
                coord=np.mean(atom_coords,axis=0)
                coords.append(coord)
        for i, coord_i in enumerate(coords):
            for j, coord_j in enumerate(coords):
                if j<=i:
                    continue
                diff_vector=coord_i-coord_j
                d=np.sqrt(np.sum(diff_vector * diff_vector))
                if  d is not None and d<= 14:
                    left.append(i)
                    right.append(j)
                    #双向边
                    left.append(j)
                    right.append(i)
                    weight = np.log(abs(i-j))/d
                    edge_attr.append([weight])
                    edge_attr.append([weight])

        edges.append(left)
        edges.append(right)
        return edges, edge_attr, coords 
    
    def _get_surface_mask(resdepth_file, surface_threshold):
        with open(resdepth_file,'r') as f:
            surface_mask = [1 if float(e.split()[1]) <surface_threshold else 0 for e in f.readlines()]
        return surface_mask

class Annotate:
    
    def annotate_binding_sites(args):
        
        parser = PDBParser()
        inputs=os.listdir(args.input_dir)
        AA_dic = {'GLY':'G','ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','TYR':'Y','ASP':'D','ASN':'N',
          'GLU':'E','LYS':'K','GLN':'Q','MET':'M','SER':'S','THR':'T','CYS':'C','PRO':'P','HIS':'H','ARG':'R'}
        AA = ['GLY','ALA','VAL','LEU','ILE','PHE','TRP','TYR','ASP','ASN',
            'GLU','LYS','GLN','MET','SER','THR','CYS','PRO','HIS','ARG']
        NA = ['DA', 'DC', 'DT', 'DG','A', 'C', 'T', 'U', 'G']
        
        for complex_file in inputs:
            complex_path=os.path.join(args.input_dir,complex_file)
            filename,ext=os.path.splitext(complex_file)
            if ext==".cif":
                parser = MMCIFParser()
            structure=parser.get_structure(filename,complex_path)
            if not structure:
                raise InvalidStructureError(filename)
            for model in structure:
                output_dir=args.output_dir
                binding_sites={}
                atom_list=[atom for atom in model.get_atoms() if atom.parent.id[0]==' ']
                searcher = NeighborSearch(atom_list=atom_list, bucket_size=10)
                contac = searcher.search_all(radius=args.annotate_threshold, level='R')
                for e in contac:
                    if (str(e[0].resname).strip() in AA) and (str(e[1].resname).strip() in NA):
                        AA_chain=e[0].parent.id
                        NA_chain=e[1].parent.id
                        record=AA_dic[e[0].resname]+str(e[0].id[1])

                        if AA_chain not in binding_sites:
                            binding_sites[AA_chain] = {}

                        if NA_chain not in binding_sites[AA_chain]:
                            binding_sites[AA_chain][NA_chain] = []

                        if record not in binding_sites[AA_chain][NA_chain]: #一个残基可能结合多个碱基，因此会重复出现
                            binding_sites[AA_chain][NA_chain].append(record)

                    if(str(e[0].resname).strip() in NA) and (str(e[1].resname).strip() in AA):
                        AA_chain=e[1].parent.id
                        NA_chain=e[0].parent.id
                        record=AA_dic[e[1].resname]+str(e[1].id[1])

                        if AA_chain not in binding_sites:
                            binding_sites[AA_chain] = {}

                        if NA_chain not in binding_sites[AA_chain]:
                            binding_sites[AA_chain][NA_chain] = []

                        if record not in binding_sites[AA_chain][NA_chain]:
                            binding_sites[AA_chain][NA_chain].append(record)
                            
                if not binding_sites:
                    raise NotPNAComplexError(filename)

                AA_chains=sorted(binding_sites.keys())
                result=''
                for AA_chain in AA_chains:
                    binding_dict=binding_sites[AA_chain]
                    NA_chains=sorted(binding_dict.keys())

                    tmp=f'{filename}:{AA_chain}:'+'_'.join(NA_chains)+'\t'

                    for NA_chain in NA_chains:
                        binding_list=binding_dict[NA_chain]
                        binding_list.sort(key=lambda x: int(x[1:]))
                        tmp+=f'{AA_chain}_{NA_chain} '+' '.join(binding_list)+':'
                        
                    tmp=tmp[:-1]
                    result+=tmp+'\n'

                tmp_path=os.path.join(output_dir,f'{filename}_tmp.txt')
                with open(tmp_path,'w') as f:
                    f.write(result)

                if len(structure)>1:
                    save_path=os.path.join(output_dir,f'{filename}(model{model.id}).txt')
                else:
                    save_path=os.path.join(output_dir,f'{filename}.txt')

                os.rename(tmp_path,save_path)

    def run_ours(args):

        model=MyModel(6,6)
        model.load_state_dict(torch.load("/home/hzeng/p/NBR_Pred/saved_model/hipbyd94_model_20-epoch.pth"))
        model.eval()
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device=torch.device('cpu')
        model.to(device)

        inputs=os.listdir(args.input_dir)
        prot_list=[]
        for pdb_file in inputs:
            filename,ext=os.path.splitext(pdb_file)
            prot_list.append(filename)
            pdb_path=os.path.join(args.input_dir,pdb_file)
            fasta_path=os.path.join(args.output_dir,f'{filename}.fasta')
            
            FeatExtract.run_esm2(args)
            FeatExtract.run_saprot(args)
            FeatExtract._get_seq(pdb_path,args.output_dir)
            FeatExtract._run_BLAST(fasta_path,filename,args.output_dir)
            FeatExtract._run_HHblits(fasta_path,filename,args.output_dir)
            FeatExtract._get_resdepth(pdb_path,filename,args.output_dir)
        
            hmm=torch.from_numpy(FeatExtract._get_hmm(os.path.join(args.output_dir,f'{filename}.hhm'))).to('cpu')
            pssm=torch.from_numpy(FeatExtract._get_pssm(os.path.join(args.output_dir,f'{filename}.pssm'))).to('cpu')
            seq_rep=torch.load(os.path.join(args.output_dir,f'{filename}.esm2'),map_location='cpu')
            struct_rep=torch.load(os.path.join(args.output_dir,f'{filename}.saprot'),map_location='cpu')
            edges, edge_attr,coords=FeatExtract._get_edges(pdb_path)
            coords=torch.tensor(coords).float()
            edges=torch.tensor(edges).int16()
            edge_attr=torch.tensor(edge_attr).float()
            surface_mask=FeatExtract._get_surface_mask(os.path.join(args.output_dir,f'{filename}.resdepth'),2.1)
            surface_mask=torch.Tensor(surface_mask).bool()
            assert hmm.shape[0]==pssm.shape[0]==seq_rep.shape[0]==struct_rep.shape[0]==len(surface_mask)==len(coords)
            node_features=torch.cat([hmm,pssm,seq_rep,struct_rep],dim=1).float()

            with torch.no_grad():
                outputs = model(node_features,coords,edges,edge_attr,surface_mask)
                outputs=outputs.detach().cpu().numpy()
                np.savetxt(os.path.join(args.output_dir,f'{filename}.txt'),outputs,delimiter=',',fmt='%.6f')

            os.remove(fasta_path)
            os.remove(os.path.join(args.output_dir,f'{filename}.hhm'))
            os.remove(os.path.join(args.output_dir,f'{filename}.pssm'))
            os.remove(os.path.join(args.output_dir,f'{filename}.esm2'))
            os.remove(os.path.join(args.output_dir,f'{filename}.saprot'))
            os.remove(os.path.join(args.output_dir,f'{filename}.resdepth'))

class Map:

    def _calculate_distance(res_i, res_j):
        res_i_coord = res_i['CA'].coord if 'CA' in res_i else np.mean([atom.coord for atom in res_i], axis=0)
        res_j_coord = res_j['CA'].coord if 'CA' in res_j else np.mean([atom.coord for atom in res_j], axis=0)
        
        diff_vector = res_i_coord - res_j_coord
        return np.sqrt(np.sum(diff_vector * diff_vector))

    def get_distance_map(args):

        parser = PDBParser()
        inputs=os.listdir(args.input_dir)
        for pdb_file in inputs:
            pdb_path=os.path.join(args.input_dir,pdb_file)
            filename,ext=os.path.splitext(pdb_file)

            if ext==".cif":
                parser = MMCIFParser()

            structure=parser.get_structure(filename,pdb_path)
            for model in structure:
                output_dir=args.output_dir

                residues=[residue for residue in model.get_residues() if residue.id[0]==' '] #标准残基
                seq=','.join([f'{residue.resname} {residue.id[1]}' for residue in residues])
                res_num = len(residues)
                dist_map = np.zeros((res_num, res_num))
            
                for i in range(res_num):
                    for j in range(i, res_num):
                        dist_map[i][j] = dist_map[j][i] = Map._calculate_distance(residues[i], residues[j])
                if args.map_d_threshold>0:
                    dist_map = np.where(dist_map < args.map_d_threshold, 0, 1)

                tmp_path=os.path.join(output_dir,f'{filename}_tmp.csv')
                if len(structure)>1:
                    save_path=os.path.join(output_dir,f'{filename}(model_{model.id}).csv')
                else:
                    save_path=os.path.join(output_dir,f'{filename}.csv')

                if args.map_d_threshold>0:
                    np.savetxt(tmp_path, dist_map, delimiter=',', header=seq, comments='', fmt='%d')
                else:
                    np.savetxt(tmp_path, dist_map, delimiter=',', header=seq, comments='')

                os.rename(tmp_path,save_path)

    def get_knn_map(args):

        parser = PDBParser()
        inputs=os.listdir(args.input_dir)
        for pdb_file in inputs:
            pdb_path=os.path.join(args.input_dir,pdb_file)
            filename,ext=os.path.splitext(pdb_file)


            if ext==".cif":
                parser = MMCIFParser()

            structure=parser.get_structure(filename,pdb_path)
            for model in structure:
                output_dir=args.output_dir

                residues=[residue for residue in model.get_residues() if residue.id[0]==' '] #标准残基
                seq=','.join([f'{residue.resname} {residue.id[1]}' for residue in residues])
                res_num = len(residues)
                dist_map = np.zeros((res_num, res_num))
                for i in range(res_num):
                    for j in range(i, res_num):
                        dist_map[i][j] = dist_map[j][i] = Map._calculate_distance(residues[i], residues[j])
                
                knn_map = np.zeros((res_num, res_num))
                for i in range(res_num):
                    dist_map[i][i] = np.inf  # 排除自身
                    knn_indices = np.argsort(dist_map[i])[:args.map_k_num]
                    knn_map[i, knn_indices] = 1
                    knn_map[knn_indices, i] = 1
                
                tmp_path=os.path.join(output_dir,f'{filename}_tmp.csv')
                if len(structure)>1:
                    save_path=os.path.join(output_dir,f'{filename}(model_{model.id}).csv')
                else:
                    save_path=os.path.join(output_dir,f'{filename}.csv')
                np.savetxt(tmp_path, knn_map, delimiter=',', fmt='%d', header=seq, comments='')
                os.rename(tmp_path,save_path)

class Split:
    
    def split_complex(args):
        AA = ['GLY','ALA','VAL','LEU','ILE','PHE','TRP','TYR','ASP','ASN',
        'GLU','LYS','GLN','MET','SER','THR','CYS','PRO','HIS','ARG']
        DA = ['DA', 'DC', 'DT', 'DG']
        RA = ['A', 'C', 'T', 'U', 'G']
        parser = PDBParser()
        io = PDBIO()
        inputs=os.listdir(args.input_dir)
        for complex_file in inputs:
            complex_path=os.path.join(args.input_dir,complex_file)
            filename,ext=os.path.splitext(complex_file)

            if ext==".cif":
                parser = MMCIFParser()

            structure=parser.get_structure(filename,complex_path)
            for model in structure:
                output_dir=args.output_dir

                for chain in model:
                    is_protein = any(res.resname in AA for res in chain)
                    is_dna = any(res.resname in DA for res in chain)
                    is_rna = any(res.resname in RA for res in chain)

                    chain_structure = Structure.Structure(chain.id)
                    chain_structure.add(model.__class__(model.id))
                    chain_structure[0].add(chain)

                    io.set_structure(chain_structure)
                    tmp_path=os.path.join(output_dir,f'{filename}_{chain.id}_tmp.pdb')
                    if len(structure)>1:
                        save_path=os.path.join(output_dir,f'{filename}_{chain.id}(model{model.id}).pdb')
                    else:
                        save_path=os.path.join(output_dir,f'{filename}_{chain.id}.pdb')
                    io.save(tmp_path)
                    os.rename(tmp_path,save_path)

    def extract_fragment(args):

        parser=PDBParser()
        io = PDBIO()
        inputs=os.listdir(args.input_dir)
        for pdb_file in inputs:
            pdb_path=os.path.join(args.input_dir,pdb_file)
            filename,ext=os.path.splitext(pdb_file)

            if ext==".cif":
                parser = MMCIFParser()

            structure=parser.get_structure(filename,pdb_path)
            for model in structure:
                output_dir=args.output_dir
                
                if args.chain_id not in model:
                    chains=",".join(model.child_dict.keys())
                    raise ChainNotFoundError(filename,chains,args.chain_id)

                chain=model[args.chain_id]
                res_indices=[res.id[1] for res in chain if res.id[0]==' ']
                min_index, max_index=min(res_indices),max(res_indices)
                if args.split_start<min_index or args.split_end>max_index:
                    raise ResIndexOutOfRangeError(filename,args.chain_id,f'{min_index}-{max_index}',f'{args.split_start}-{args.split_end}')

                io.set_structure(chain)

                tmp_path=os.path.join(output_dir,f'{filename}_({args.split_start}-{args.split_end})_tmp.pdb')

                if len(structure)>1:
                    save_path=os.path.join(output_dir,f'{filename}_{args.chain_id}_({args.split_start}-{args.split_end})(model{model.id}).pdb')
                else:
                    save_path=os.path.join(output_dir,f'{filename}_{args.chain_id}_({args.split_start}-{args.split_end}).pdb')

                io.save(tmp_path,ResidueRangeSelect(args.split_start,args.split_end))
                os.rename(tmp_path,save_path)

def cluster(args):
    os.system(f"{FOLDSEEK} easy-cluster ")
    pass

def download_pdb(prot_list,save_path):
    pdbl = PDBList()
    for prot in prot_list:
        pdbl.retrieve_pdb_file(prot,pdir=save_path)

def main(config):
    args=Args(**config)
    if args.prot_id:
        download_pdb(args.prot_id.split(','),args.input_dir)

    match args.task_name:
        case "esm2":
            FeatExtract.run_esm2(args)
        case "saprot":
            FeatExtract.run_saprot(args)
        case "annotate_complex":
            Annotate.annotate_binding_sites(args)
        case "d_map":
            Map.get_distance_map(args)
        case "knn_map":
            Map.get_knn_map(args)
        case "split_complex":
            Split.split_complex(args)
        case "extract_fragment":
            Split.extract_fragment(args)
        case "handcraft":
            FeatExtract.get_handcraft_features(args)
        case "nbrpred":
            Annotate.run_ours(args)

if __name__=="__main__":
    args=tyro.cli(Args)
    args=Args(
        input_dir="./task/uploads/test",
        output_dir="./task/results/test",
        task_name="annotate_complex",
        annotate_threshold=4.5,
        chain_id="z",
        split_start=1006,
        split_end=1020
        )
    match args.task_name:
        case "esm2":
            FeatExtract.run_esm2(args)
        case "saprot":
            FeatExtract.run_saprot(args)
        case "annotate_complex":
            Annotate.annotate_binding_sites(args)
        case "d_map":
            Map.get_distance_map(args)
        case "knn_map":
            Map.get_knn_map(args)
        case "split_complex":
            Split.split_complex(args)
        case "extract_fragment":
            Split.extract_fragment(args)
        case "handcraft":
            FeatExtract.get_handcraft_features(args)
        case "nbrpred":
            Annotate.run_ours(args)