import os

class Builder:
    def __init__(self):
        self.results={}
        
    def add(self,name,**kwargs):
        self.results.setdefault(name,[]).append(kwargs)

    def get(self):
        return self.results
    
def annoatation_results(task_id):
    builder=Builder()
    output_dir=f'./task/results/{task_id}'
    files=os.listdir(output_dir)
    for file in files:
        filename,ext=os.path.splitext(file)
        if ext=='.txt':
            with open(os.path.join(output_dir,file)) as f:
                lines=f.readlines()
                for line in lines:
                    line=line.strip()
                    if line:
                        part1,part2=line.split('\t')
                        name=part1.split(':')[0]
                        for item in part2.split(':'):
                            chains=item.split(' ')[0]
                            AA_chain=chains.split('_')[0]
                            NA_chain=chains.split('_')[1]
                            residues=' '.join(item.split(' ')[1:])
                            builder.add(name,AA_chain=AA_chain,NA_chain=NA_chain,binding_residues=residues)
    return builder.get()

def map_results(task_id):
    builder=Builder()
    output_dir=f'./task/results/{task_id}'
    files=os.listdir(output_dir)
    for file in files:
        filename,ext=os.path.splitext(file)
        if ext=='.csv':
            with open(os.path.join(output_dir,file)) as f:
                head=f.readline()
                content=f.readlines()
            head=head.strip().split(',')
            n=len(head)
            upper_triangle=[]
            for i,line in enumerate(content):
                eliments=line.split(',')
                for j in range(i,n):
                    upper_triangle.append(float(eliments[j]))

            builder.add(filename,head=head,upper_triangle=upper_triangle)

    return builder.get()

def fragment_results(task_id):
    builder=Builder()
    output_dir=f'./task/results/{task_id}'
    files=os.listdir(output_dir)
    for file in files:
        filename,ext=os.path.splitext(file)
        if ext=='.pdb':
            with open(os.path.join(output_dir,file)) as f:
                for line in f:
                    if line.startswith("ATOM"):
                        builder.add(
                            filename,
                            record=line[0:6],
                            atomNumber=line[6:11],
                            atomName=line[12:16],
                            altLoc=line[16:17],
                            resName=line[17:20],
                            chainID=line[21:22],
                            resSeq=line[22:26],
                            iCode=line[26:27],
                            x=line[30:38],
                            y=line[38:46],
                            z=line[46:54],
                            occupancy=line[54:60],
                            tempFactor=line[60:66],
                            element=line[76:78],
                            charge=line[78:80]
                        )
    return builder.get()

def handcraft_results(task_id):
    pass

def nbrpred_results(task_id):
    pass

def read(task_name,task_id):
    match task_name:
        case "annotate_complex":
            return annoatation_results(task_id)
        case "d_map":
            return map_results(task_id)
        case "knn_map":
            return map_results(task_id)
        case "extract_fragment":
            return fragment_results(task_id)
        case "handcraft":
            return handcraft_results(task_id)
        case "nbrpred":
            return nbrpred_results(task_id)
    
if __name__=="__main__":
    results=read("annotate_complex","173158896392127615746")
   
    print(results)