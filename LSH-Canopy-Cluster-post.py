import numpy as np
####################################################################
import collections
import os
import stat
import subprocess
import time
import gc
import shutil 
from shutil import copyfile
import errno
import math
from bisect import bisect_left
import sys
####################################################################
import linecache
####################################################################
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue
####################################################################

domain=['A','C','G','T']
kmerIndex = {}
kIndex = 0
for i in range(len(domain)):
    for j in range(len(domain)):
        for k in range(len(domain)):
            for l in range(len(domain)):
                kmerPermutation=domain[i]+domain[j]+domain[k]+domain[l]
                kmerPermutation=kmerPermutation.strip()
                kmerIndex[kmerPermutation]=kIndex
                kIndex+=1

#####################################################################

#Paths
copyToTemp=True
#temp_dir=os.path.join('/tmp/mrahma23/')
#data_dir=os.path.join('/data/scratch/mrahma23/data/')
#processed_data_dir=os.path.join('/data/scratch/mrahma23/processed_data/')

data_dir=os.getcwd()
temp_dir=os.path.join(os.getcwd(),'tmp')
processed_data_dir=os.path.join(os.getcwd(),'Processed_Data')

#LSH weights
d=200
randv = np.random.randn(d, 256)

#For canopies
# t1=0.45
# t2=0.25
maxMembersInCanopy=99999999999999999999999999

#####################################################################

def get_signature(user_vector): 
    res = 0
    val = np.dot(randv,user_vector)
    for v in val:
        res = res << 1
        if v >= 0:
            res |= 1
    return res

#######################################################################################################################################################

def kmer_list(dna, k):
    result = []
    for x in range(len(dna)+1-k):
        result.append(dna[x:x+k])
    return result

#######################################################################################################################################################

def readFile(fileName,inq):    
    if copyToTemp==True:
        completeDataFile=os.path.join(temp_dir,fileName)
    else:
        completeDataFile=os.path.join(data_dir,fileName)
    numberOfInstances=0
    f = open(completeDataFile, 'r')
    sequence=''
    meta=''
    for line in f:
        line=line.strip()
        if line!='' and line!='\n' and line!=' ':
            if line.startswith('>'):
                if sequence!='':
                    item=(numberOfInstances,sequence)
                    inq.put(item)
                    numberOfInstances+=1
                    sequence=''
            else:
                sequence=sequence+line    
    if sequence!='':
        item=(numberOfInstances,sequence)
        inq.put(item)
        numberOfInstances+=1
        sequence=''    
    f.close()

#######################################################################################################################################################

def fastaToNumeric(inq,outq):
    while True:
        item=inq.get()
        if item==None:
            break
        else:
            sampleStat=[0.0]*256
            sequenceNumber=item[0]
            sequence=item[1]
            klist=kmer_list(sequence,4)
            c=collections.Counter(klist)
            for kmer in c.elements():
                otherCharacters=False
                for character in kmer:
                    if character not in 'ACGT':
                        otherCharacters=True
                        break
                if otherCharacters==False: 
                    sampleStat[kmerIndex[kmer]]=c[kmer]
            minSampleStat=min(sampleStat)
            maxSampleStat=max(sampleStat)
            sampleStat=[(float)((i-minSampleStat)/(maxSampleStat-minSampleStat)) for i in sampleStat]
            codeLSH=str(sequenceNumber)+'-'+str(get_signature(sampleStat))+'\n'
            outq.put(codeLSH)

#######################################################################################################################################################    

def write_LSH_codes(fileName,outq):
    if copyToTemp==True:
        LSHDest=os.path.join(temp_dir,fileName.split('.')[0] + '_processed_LSH.txt')
        numOfInstances=os.path.join(temp_dir,fileName.split('.')[0] + '_total_instances.txt')
    else:
        LSHDest=os.path.join(processed_data_dir,fileName.split('.')[0] + '_processed_LSH.txt')
        numOfInstances=os.path.join(processed_data_dir,fileName.split('.')[0] + '_total_instances.txt')
            
    totalNumberOfInstances=0
    with open(LSHDest, 'w') as f:
        while True:
            line=outq.get()
            if line==None:
                break
            else:
                f.write(line)
                totalNumberOfInstances+=1    
    with open(numOfInstances,'w') as f:
        f.write(str(totalNumberOfInstances))    
    
#     if copyToTemp==True:
#         destination=os.path.join(processed_data_dir,fileName.split('.')[0] + '_processed_LSH.txt')
#         source=os.path.join(temp_dir,fileName.split('.')[0] + '_processed_LSH.txt')
#         copyfile(source, destination)
#         destination=os.path.join(processed_data_dir,fileName.split('.')[0] + '_total_instances.txt')
#         source=os.path.join(temp_dir,fileName.split('.')[0] + '_total_instances.txt')
#         copyfile(source, destination)

#######################################################################################################################################################


def canopyCluster(fileName, queue, t1, t2): 
    output = '---------------------------------------------------------------------------------------' + '\n' + 'Starting Canopy Clustering for: ' + fileName + '\n'
    start=time.time()    
    if copyToTemp==True:
        destination=os.path.join(temp_dir,fileName.split('.')[0] + '_processed_LSH.txt')  
        numberOfInstances=os.path.join(temp_dir,fileName.split('.')[0] + '_total_instances.txt')
        canopyMemberpath=os.path.join(temp_dir,fileName.split('.')[0] + '_canopy_memberships.txt')
    else:
        destination=os.path.join(processed_data_dir,fileName.split('.')[0] + '_processed_LSH.txt')
        numberOfInstances=os.path.join(processed_data_dir,fileName.split('.')[0] + '_total_instances.txt')
        canopyMemberpath=os.path.join(processed_data_dir,fileName.split('.')[0] + '_canopy_memberships.txt')    
    totalNumberOfInstance=0
    with open(numberOfInstances,'r') as f:
        for line in f: 
            if line!='' and line!=' ' and line!='\n':
                totalNumberOfInstance=(int)(line.strip())    
    bitmap=np.ones(totalNumberOfInstance)    
    instanceLSH=[0]*totalNumberOfInstance
    with open(destination,'r') as f:
        for line in f:
            if line!='' and line!=' ' and line!='\n':
                items=(line.strip()).split('-')
                instanceNumber=(int)(items[0])
                LSHCode=(int)(items[1])
                instanceLSH[instanceNumber]=LSHCode       
    numberOfCanopies=0
    numberOfInstancesInAllCanopies=0
    worstCaseCalculation=0
    with open(canopyMemberpath,'w') as f:
        while np.nonzero(bitmap)[0].size!=0:
            canopyIndex=(int)(np.random.choice(np.nonzero(bitmap)[0],1))
            canopy=instanceLSH[canopyIndex]
            numberOfCanopies+=1
            canopyMembers=[]            
            for instanceNumber, instanceSignature in enumerate(instanceLSH,0):
                if bitmap[instanceNumber]>0:
                    xor=instanceSignature^canopy
                    hash_measure=(bin(xor)[2:].count('1'))/float(d)
                    if hash_measure<=t2:
                        canopyMembers.append(instanceNumber)
                        bitmap[instanceNumber]=0
                    elif hash_measure<=t1:
                        canopyMembers.append(instanceNumber)
                if len(canopyMembers)>=maxMembersInCanopy:
                    break            
            numberOfInstancesInAllCanopies=numberOfInstancesInAllCanopies+len(canopyMembers)
            worstCaseCalculation=worstCaseCalculation+(len(canopyMembers)*len(canopyMembers))
            for item in canopyMembers:
                f.write(str(item)+' ')
            f.write('\n')
            queueContent=(numberOfCanopies,canopyMembers)
            queue.put(queueContent)
    
    duration=time.time()-start
    output = output +  '---------------------------------------------------------------------------------------' + '\n' + 'Number of Canopies: ' + str(numberOfCanopies) + ' with T1: ' + str(t1) + ' and T2: ' + str(t2) + '\n' + 'Total Number of Instances: ' + str(totalNumberOfInstance) + '\n' + 'Total Number of Instances in All Canopies: ' + str(numberOfInstancesInAllCanopies) + '\n' + '---------------------------------------------------------------------------------------' + '\n' + 'Worst-case calculations without canopy: ' + str(totalNumberOfInstance) + '*' + str(totalNumberOfInstance) + '=' + str(totalNumberOfInstance*totalNumberOfInstance) + '\n' + 'Worst-case calculations with canopy: ' + str(worstCaseCalculation) + '\n' + 'Difference in worst case calculations: ' + str((totalNumberOfInstance*totalNumberOfInstance)-worstCaseCalculation) + '\n' + '---------------------------------------------------------------------------------------' + '\n' + 'Time for Canopy Clustering: ' + str(duration) + ' Seconds or ' + str(duration/60) + ' Minutes' + '\n'
#     if copyToTemp==True:
#         dest=os.path.join(processed_data_dir,fileName.split('.')[0] + '_canopy_memberships.txt')
#         copyfile(canopyMemberpath, dest)
    print output
    return output

#######################################################################################################################################################

def binary_search(a, x, lo=0):   
    hi = len(a)   
    pos = bisect_left(a,x,lo,hi)
    if pos != hi and a[pos] == x:
        return True       
    else:
        return False 

#######################################################################################################################################################

def splitFile(fileName, queue, expQue):
    while True:
        item=queue.get()        
        if item==None:
            break        
        canopyID=item[0]
        listOfInstances=item[1]        
        if copyToTemp==True:     
            canopyFolder=os.path.join(temp_dir, fileName.split('.')[0] + '_canopy_files')            
            try:
                os.makedirs(canopyFolder)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise            
            completeDataFile=os.path.join(temp_dir,fileName)
            canopyFilePath=os.path.join(canopyFolder, fileName.split('.')[0] + '-' + str(canopyID) + '.fasta')
        else:
            completeDataFile=os.path.join(data_dir,fileName)
            canopyFilePath=os.path.join(processed_data_dir, fileName.split('.')[0] + '-' + str(canopyID) + '.fasta')        
        numberOfInstances=0
        maxCanopyMembers=len(listOfInstances)
        maxIndex=max(listOfInstances)
        sequence=''
        meta=''
        f = open(completeDataFile, 'r')
        with open(canopyFilePath, 'w') as c:    
            for line in f:
                if line!='' and line!='\n' and line!=' ':
                    if line.startswith('>'):
                        if meta!='' and sequence!='':                            
                            found=binary_search(listOfInstances, numberOfInstances)
                            if found==True:
                                c.write(meta)
                                c.write(sequence)
                            numberOfInstances+=1
                            sequence=''
                            if numberOfInstances>maxIndex:
                                break                            
                        meta=line
                    else:
                        sequence=sequence+(line.replace('N',''))
                else:
                    if meta!='' and sequence!='' and numberOfInstances<=maxIndex:
                        found=binary_search(listOfInstances, numberOfInstances)
                        if found==True:
                            c.write(meta)
                            c.write(sequence)
                        numberOfInstances+=1
                        sequence=''
        f.close()
        expQue.put(fileName.split('.')[0] + '-' + str(canopyID) + '.fasta')

#######################################################################################################################################################

def expensiveCluster(que, joinQueue, method):
    while True:
        gc.collect()
        smallFileName=que.get()
        if smallFileName==None:
            break
        canopyID=smallFileName.split('.')[0].split('-')[1]
        fileName=smallFileName.split('.')[0].split('-')[0]        
        if copyToTemp==True:    
            canopyFolder=os.path.join(temp_dir, fileName + '_canopy_files')
        else:
            canopyFolder=os.path.join(processed_data_dir)       
        
        inputFile = smallFileName
                
        commandPrefix = 'pick_otus.py -i ' + inputFile + ' -o ' + smallFileName.split('.')[0] + ' -m ' + method 
        
        os.chdir(canopyFolder)
        returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
        
        source=os.path.join(smallFileName.split('.')[0], smallFileName.split('.')[0] + '_otus.txt')
        destination=os.path.join(smallFileName.split('.')[0] + '_otus.txt')
        
        if os.path.isfile(source)==True:
            copyfile(source,destination)
            deleteFile=os.path.join(canopyFolder,smallFileName)
            os.remove(deleteFile)
            joinQueue.put(smallFileName.split('.')[0] + '_otus.txt')    
        shutil.rmtree(smallFileName.split('.')[0])
            
#######################################################################################################################################################


def joinClusteredOutput(fileName,que):
    if copyToTemp==True:      
        canopyFolder=os.path.join(temp_dir, fileName.split('.')[0] + '_canopy_files')
        canopyFilePath=os.path.join(temp_dir, fileName.split('.')[0] + '_canopy_files', fileName.split('.')[0] + '_temp_otus.txt')
    else:
        canopyFolder=os.path.join(processed_data_dir)
        canopyFilePath=os.path.join(processed_data_dir, fileName.split('.')[0] + '_temp_otus.txt')
    
    previousOTU={}
    
    with open(canopyFilePath, 'w') as c:        
        while True:
            smallFileName=que.get()
            if smallFileName==None:
                break
            smallClusteredFile=os.path.join(canopyFolder,smallFileName)
            with open(smallClusteredFile,'r') as f:
                contents=f.readlines()
            for item in contents:
                c.write(item)                
                
                otu=item.strip().split()[0]
                otuMembers=item.strip().split()[1:]
                previousOTU[otu]=otuMembers                
            
            os.remove(smallClusteredFile)    
    
    if copyToTemp==True:
        destination=os.path.join(processed_data_dir, fileName.split('.')[0] + '_temp_otus.txt')
        copyfile(canopyFilePath, destination)
        gc.collect()
    
    #Pick rep
    print 'Now Creating Merging Rep for clusters...'
    dataFile=os.path.join(data_dir,fileName)
    tempOutputFile=os.path.join(processed_data_dir,fileName.split('.')[0] + '_temp_rep.fna')
    commandPrefix = 'pick_rep_set.py -i ' + destination + ' -f ' + dataFile + ' -o ' + tempOutputFile
    returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print 'Return Code for temporary Merging Rep picking: ' + str(returnCode) + '\n'
    
    #Merging UCLUST
    inputFile = tempOutputFile
    mergeOutput = os.path.join(processed_data_dir, 'tmp') 
    commandPrefix = 'pick_otus.py -i ' + inputFile + ' -o ' + mergeOutput + ' -m uclust -s 0.99'
    returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print 'Return Code for Merging UCLUST: ' + str(returnCode) + '\n'
    
    source=os.path.join(mergeOutput, fileName.split('.')[0] + '_temp_rep_'+ 'otus.txt')
    destination=os.path.join(processed_data_dir, fileName.split('.')[0] + '_temp_rep_'+ 'otus.txt')
    copyfile(source, destination)
    shutil.rmtree(mergeOutput)
    
    with open(destination,'r') as f:
        newOTU=f.readlines()
    
    OTUcount=0
    singleton=0
    otuFile=os.path.join(processed_data_dir,fileName.split('.')[0] + '_otus.txt')   
    with open(otuFile,'w') as w:
        for item in newOTU:
            otuID=item.strip().split()[0]
            newOtuMembers=item.strip().split()[1:]
            replaceMembers=set()
            for member in newOtuMembers:
                oldReads=previousOTU[member]
                for read in oldReads:
                    replaceMembers.add(read)
            if len(replaceMembers)>=2:
                w.write(otuID+'\t')
                for rd in replaceMembers:
                    w.write(rd + '\t')
                w.write('\n')
                OTUcount+=1
            else:
                singleton+=1
    with open(os.path.join(processed_data_dir,'count'),'w') as f:
        f.write('Total OTU (Non-Singleton)' + str(OTUcount) + '\n' + 'Total SIngleton: ' + str(singleton))    
        
        
#######################################################################################################################################################

def postProcess(fileName, method, k=1, measureScore=False):
    ##################################################################################################################################
    if k==0:
        referenceTaxaMap=os.path.join(data_dir,'97_Silva_111_taxa_map_RDP_6_levels.txt')
        referenceTaxaRep=os.path.join(data_dir,'97_Silva_111_rep_set.fasta')
    else:
        referenceTaxaMap=os.path.join(data_dir,'97_otu_taxonomy.txt')
        referenceTaxaRep=os.path.join(data_dir,'97_otus.fasta')
    ##################################################################################################################################
    if '18s' in fileName:
        pyNastTemplate='core_Silva119_alignment.fna'
    else:
        pyNastTemplate='core_set_aligned.fasta.imputed'  
    ##################################################################################################################################
    dataFile=os.path.join(data_dir,fileName)
    otuFile=os.path.join(processed_data_dir,fileName.split('.')[0] + '_otus.txt')
    tempOutputFile=os.path.join(processed_data_dir,fileName.split('.')[0] + '_rep.fna')      
    ##################################################################################################################################
    #Pick rep
    print '1. Creating final Rep for clusters...'
    commandPrefix = 'pick_rep_set.py -i ' + otuFile + ' -f ' + dataFile + ' -o ' + tempOutputFile
    returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print 'Return Code for Rep picking: ' + str(returnCode) + '\n'
    ##################################################################################################################################
    #assign Tax
    print '2. Assigning Taxonomy from provided database...'
    print 'Name of Taxa Database: ' + referenceTaxaRep
    print 'Name of Taxa Map: ' + referenceTaxaMap
    taxonomyDir=os.path.join(processed_data_dir,'Taxonomy')
    commandPrefix = 'assign_taxonomy.py -i ' + tempOutputFile + ' -t ' + referenceTaxaMap + ' -r ' + referenceTaxaRep + ' -m rdp -c 0.80 -o ' + taxonomyDir
    returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print 'Return Code for Taxonomy Assignment: ' + str(returnCode) + '\n'    
    if returnCode==0: 
        source=os.path.join(processed_data_dir, 'Taxonomy', fileName.split('.')[0] + '_rep_tax_assignments.txt')
        destination=os.path.join(processed_data_dir, fileName.split('.')[0] + '_rep_tax_assignments.txt')
        copyfile(source, destination)
        shutil.rmtree(os.path.join(processed_data_dir,'Taxonomy'))
    else:
        print 'Error! Possible Reasons: RDP path wasnt set, OTU rep wasnt generated, Qiime wasnt setup successfully etc....'    
    ##################################################################################################################################    
    #Doing PyNast Alignment
    print '3. Now Aligning...(with PyNast)'
    inputUnAligned=os.path.join(processed_data_dir, fileName.split('.')[0] + '_rep.fna')
    referenceAligned=os.path.join(data_dir,pyNastTemplate)
    print 'Name of PyNast Template File: ' + referenceAligned
    outputAligned=os.path.join(processed_data_dir, 'aligned')
    commandPrefix = 'parallel_align_seqs_pynast.py -i ' + inputUnAligned + ' -t ' + referenceAligned + ' -o ' + outputAligned + ' -O 8'   
    #commandPrefix = 'align_seqs.py -i ' + inputUnAligned + ' -t ' + referenceAligned + ' -o ' + outputAligned
    returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print 'Return Code for Alignment: ' + str(returnCode) + '\n'
    source=os.path.join(processed_data_dir, 'aligned',fileName.split('.')[0] + '_rep_aligned.fasta')
    destination=os.path.join(processed_data_dir,fileName.split('.')[0] + '_rep_aligned.fasta')
    copyfile(source, destination)
    ###################################################################################################################################
    print '4. Now filtering alignment...'
    inputAligned=os.path.join(processed_data_dir, 'aligned',fileName.split('.')[0] + '_rep_aligned.fasta')
    outputFilteredAligned=os.path.join(processed_data_dir, 'aligned')
    commandPrefix = 'filter_alignment.py -i ' + inputAligned + ' -o ' + outputFilteredAligned
    returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print 'Return Code for filtering Alignment: ' + str(returnCode) + '\n'
    source=os.path.join(processed_data_dir, 'aligned',fileName.split('.')[0] + '_rep_aligned_pfiltered.fasta')
    destination=os.path.join(processed_data_dir,fileName.split('.')[0] + '_rep_aligned_pfiltered.fasta')
    copyfile(source, destination)
    shutil.rmtree(os.path.join(processed_data_dir, 'aligned'))
    ################################################################################################################################## 
    #Making Phylogeny Tree
    print '5. Creating Phylogeny Tree...'
    inputAlignedFiltered=os.path.join(processed_data_dir, fileName.split('.')[0] + '_rep_aligned_pfiltered.fasta')
    outputTree=os.path.join(processed_data_dir,fileName.split('.')[0] + '_rep_phylo.tre')
    commandPrefix = 'make_phylogeny.py -i ' + inputAlignedFiltered + ' -o ' +  outputTree
    returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print 'Return Code for Phylogeny Tree: ' + str(returnCode) + '\n'    
    ################################################################################################################################## 
    #Make OTU Table from OTU map and Silva Genes Taxa
    print '6. Making OTU table in BIOM format from OTU map and Taxa...'
    otuMapFile=os.path.join(processed_data_dir,fileName.split('.')[0] + '_otus.txt')
    taxaFile=os.path.join(processed_data_dir, fileName.split('.')[0] + '_rep_tax_assignments.txt')
    outputBiomFile=os.path.join(processed_data_dir,fileName.split('.')[0] + '_table.biom')
    mappingFile=os.path.join(data_dir,fileName.split('.')[0] + '_mapping_file.txt')
    commandPrefix = 'make_otu_table.py -i ' + otuMapFile + ' -t ' + taxaFile + ' -m ' + mappingFile + ' -o ' + outputBiomFile
    returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print 'Return Code for OTU table creation: ' + str(returnCode) + '\n'
    ##################################################################################################################################
    print '7. Summarize Taxa at Level 6 (Genus) ...'
    inputBiomFile=os.path.join(processed_data_dir,fileName.split('.')[0] + '_table.biom')
    outputFolder = os.path.join(processed_data_dir,fileName.split('.')[0] + '_L6_taxa_summary')
    commandPrefix = 'summarize_taxa.py -i ' + inputBiomFile + ' -L 6' + ' -o ' + outputFolder
    returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print 'Return Code for Taxa-Summary creation: ' + str(returnCode) + '\n'
    if returnCode==0: 
        source=os.path.join(outputFolder, fileName.split('.')[0] + '_table_L6.txt')
        destination=os.path.join(processed_data_dir, fileName.split('.')[0] + '_table_L6_tax_summary.txt')
        copyfile(source, destination)
        shutil.rmtree(outputFolder)
    else:
        print 'Error! Possible Reasons: RDP path wasnt set, OTU rep wasnt generated, Qiime wasnt setup successfully etc....'
    ##################################################################################################################################
    #Summarize Biom Table
    print '8. Summarize BIOM table...'
    inputBiomFIle=os.path.join(processed_data_dir,fileName.split('.')[0] + '_table.biom')
    outputBiomSUmmary=os.path.join(processed_data_dir,fileName.split('.')[0] + '_biom_summary.txt')
    commandPrefix = 'biom summarize-table -i ' + inputBiomFIle + ' --qualitative -o ' + outputBiomSUmmary
    returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print 'Return Code for BIOM-Summary creation: ' + str(returnCode) + '\n'
    ##################################################################################################################################
    #Generating Alpha Diversity
    print '9. Generating Alpha Diversity (PD_whole_tree, Shannon, Chao1, ACE)...'
    inputBiomFIle=os.path.join(processed_data_dir,fileName.split('.')[0] + '_table.biom')
    inputTreeFIle=os.path.join(processed_data_dir,fileName.split('.')[0] + '_rep_phylo.tre')
    outputAlphaDvFIle=os.path.join(processed_data_dir,fileName.split('.')[0] + '_alpha_diversity.txt')
    commandPrefix = 'alpha_diversity.py -i ' + inputBiomFIle + ' -m PD_whole_tree,shannon,chao1,ace' + ' -t ' + inputTreeFIle + ' -o ' + outputAlphaDvFIle
    returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    if returnCode!=0:
        'ACE calculation failed! Starting without ACE'
        commandPrefix = 'alpha_diversity.py -i ' + inputBiomFIle + ' -m PD_whole_tree,shannon,chao1' + ' -t ' + inputTreeFIle + ' -o ' + outputAlphaDvFIle
        returnCode=subprocess.call(commandPrefix, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print 'Return Code for Alpha Diversity: ' + str(returnCode) + '\n'   
    ##################################################################################################################################
    if measureScore==True:
        #Calculating F-Score (Expected Taxa must be provided)
        print '10. Calculating F-Score (Expected Taxa must be provided)...'
        expected_f = os.path.join(data_dir,fileName.split('.')[0] + '_expected_taxa.txt')
        expected_tax = set()
        with open(expected_f, 'r') as expected:
            for line in expected:
                if line.startswith('#'):
                    continue
                else:
                    tax = line.split()[0]
                    expected_tax.add(tax)    
        actual_f = os.path.join(processed_data_dir, fileName.split('.')[0] + '_table_L6_tax_summary.txt') 
        actual_tax = set()
        with open(actual_f, 'r') as actual:
            for line in actual:
                if line.startswith('#'):
                    continue
                else:
                    # remove ";Other" strings appended by summarize_taxa.py to extend
                    # taxonomy up to specified taxonomy level
                    if (";Other" in line) or (";other" in line):
                        line = line.replace(";Other","")
                    tax = line.strip().split("\t")[0]
                    actual_tax.add(tax)
        tp = len(actual_tax & expected_tax)
        fp = len(actual_tax - expected_tax)
        fn = len(expected_tax - actual_tax)
        p = tp / float(tp + fp)
        r = tp / float(tp + fn)
        fScore = float(2 * p * r) / float(p + r)
        with open(os.path.join(processed_data_dir,'F-measures.txt'),'w') as f:
            f.write('True Positive: ' + str(tp) + '\n' + 'False Positive: ' + str(fp) + '\n' + 'False Negative: ' + str(fn) + '\n' + 'Precision: ' + str(p) + '\n' + 'Recall: ' + str(r) + '\n' + 'F-Score: ' + str(fScore))
        
    print 'Done!'

#######################################################################################################################################################

def main(fileName,cluster_method,t1,t2):
    totalTimeStart=time.time()
    output=''
    #######################################################################################################   
    #Data Pre-Processing        
    s1 = '---------------------------------------------------------------------------------------' + '\n' + 'Starting Data Pre-Processing for: ' + fileName + '\n'
    print s1
    start=time.time()        
    
    num_workers=multiprocessing.cpu_count()
    workers=[]        
    inq = multiprocessing.Queue()
    outq = multiprocessing.Queue()
    
    for i in xrange(num_workers):
        tmp = multiprocessing.Process(target=fastaToNumeric, args=(inq,outq,))
        tmp.daemon=True
        tmp.start()
        workers.append(tmp)
    
    fileReadProcess=multiprocessing.Process(target=readFile, args=(fileName,inq,))
    fileReadProcess.daemon=True
    fileReadProcess.start()    
    
    fileWriteProcess=multiprocessing.Process(target=write_LSH_codes, args=(fileName,outq,))
    fileWriteProcess.daemon=True
    fileWriteProcess.start()
    
    fileReadProcess.join()
    if fileReadProcess.is_alive()==True:
        fileReadProcess.terminate()
        del fileReadProcess
    
    for i in xrange(num_workers):
        inq.put(None)            
    
    for worker in workers:
        worker.join()
    
    for worker in workers:
        if worker.is_alive()==True:
            worker.terminate()
            del worker          
    
    outq.put(None)        
    
    fileWriteProcess.join()        
    if fileWriteProcess.is_alive()==True:
        fileWriteProcess.terminate()
        del fileWriteProcess      
    
    inq.close()
    outq.close()
    inq.join_thread()
    outq.join_thread()
    duration=time.time()-start
    
    del num_workers
    del workers
    del inq
    del outq
    gc.collect()
    s2 = 'Time for Data Pre-processing: ' + str(duration) + ' Seconds or ' + str(duration/60) + ' Minutes' + '\n'
    print s2
    output=output+s1+s2
####################################################################################################### 
    
    #Canopy Clustering
    num_workers=multiprocessing.cpu_count()
    workers=[]        
    inq = multiprocessing.Queue()
    expensiveClusterQueue= multiprocessing.Queue()        
    
    for i in xrange(num_workers):
        tmp = multiprocessing.Process(target=splitFile, args=(fileName,inq,expensiveClusterQueue,))
        tmp.daemon=True
        tmp.start()
        workers.append(tmp)        
    
    output=output+canopyCluster(fileName,inq,t1,t2)        
    
    for i in xrange(num_workers):
        inq.put(None)          
    
    for worker in workers:
        worker.join()
    
    for worker in workers:
        if worker.is_alive()==True:
            worker.terminate()
            del worker        
    
    inq.close()
    inq.join_thread()
    del num_workers
    del workers
    del inq        
    gc.collect()
    #######################################################################################################

    #Expensive Clustering Within Canopies
    if copyToTemp==True:    
        canopyFolder=os.path.join(temp_dir, fileName.split('.')[0] + '_canopy_files')
    else:
        canopyFolder=os.path.join(processed_data_dir)
    
    finalClusterQueue= multiprocessing.Queue()
    num_workers=multiprocessing.cpu_count()
    workers=[]
    
    finalClusterWriteProcess=multiprocessing.Process(target=joinClusteredOutput, args=(fileName,finalClusterQueue,))
    finalClusterWriteProcess.daemon=True
    finalClusterWriteProcess.start()
    
    for i in xrange(num_workers):
        tmp = multiprocessing.Process(target=expensiveCluster, args=(expensiveClusterQueue, finalClusterQueue, cluster_method, ))
        tmp.daemon=True
        tmp.start()
        workers.append(tmp)
    
    for i in xrange(num_workers):
        expensiveClusterQueue.put(None)
    
    for worker in workers:
        worker.join()
    
    for worker in workers:
        if worker.is_alive()==True:
            worker.terminate()
            del worker      
    
    finalClusterQueue.put(None)
    
    finalClusterWriteProcess.join()        
    if finalClusterWriteProcess.is_alive()==True:
        finalClusterWriteProcess.terminate()
        del finalClusterWriteProcess   
    
    expensiveClusterQueue.close()
    expensiveClusterQueue.join_thread()
    finalClusterQueue.close()
    finalClusterQueue.join_thread()
    
    totalTimeDuration=time.time()-totalTimeStart
    s='---------------------------------------------------------------------------------------' + '\n' + 'Total Time for: ' + fileName + ': ' + str(totalTimeDuration) + ' Seconds or ' + str(totalTimeDuration/60) + ' Minutes'  + '\n'
    print s
    output+=s
    del num_workers
    del workers
    del expensiveClusterQueue
    del finalClusterQueue
    gc.collect()
    #######################################################################################################
   
    #Write Summary       
    if copyToTemp==True:     
        try:
            os.makedirs(temp_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        summaryOutput=os.path.join(temp_dir,fileName.split('.')[0] + '_overall_summary.txt')
    else:
        summaryOutput=os.path.join(processed_data_dir,fileName.split('.')[0] + '_overall_summary.txt')
    
    with open(summaryOutput, 'w') as f:
        f.write(output)        
    
    if copyToTemp==True:
        destination=os.path.join(processed_data_dir,fileName.split('.')[0] + '_overall_summary.txt')
        copyfile(summaryOutput, destination)
        shutil.rmtree(temp_dir)
    

#######################################################################################################################################################


if __name__ == "__main__":    
    files=['bokulich_2.fna','bokulich_3.fna','bokulich_6.fna']
    #files=['bokulich_6.fna']
    cluster_method_domain=['uclust', 'sumaclust', 'swarm']
    #cluster_method_domain=['uclust']
    ##########################################################################################################################
    taxaExpected=['bokulich_2.fna', 'bokulich_3.fna', 'bokulich_6.fna']
    noTaxa=['body_sites.fna', 'canadian_soil.fna', 'global_soil_18s.fna']
    ##########################################################################################################################    
    t1=0.45
    t2=0.35   
    for fileName in files:     
        gc.collect()
        for cluster_method in cluster_method_domain:
            gc.collect()
            print '\n\nWorking With: '
            print 'Dataset: ' + fileName
            print 'LSH-Canopy-Method: ' + cluster_method 
            try:
                os.makedirs(processed_data_dir)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise        
            shutil.rmtree(processed_data_dir)        
            try:
                os.makedirs(processed_data_dir)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise                
            subprocess._cleanup()
            gc.collect()        
            if copyToTemp==True:     
                try:
                    os.makedirs(temp_dir)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise            
                shutil.rmtree(temp_dir)            
                try:
                    os.makedirs(temp_dir)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise            
                source=os.path.join(data_dir,fileName)
                destination=os.path.join(temp_dir,fileName)
                copyfile(source, destination)       
            
            main(fileName,cluster_method,t1,t2)
            
            #Post Process, Score calculations start here
            #Arguments: Name of DataSet, Cluster Method, 0=Silva/1=GreenGenes, True=Calculate F score or False
            
            if '18s' in fileName:
                refDataset=0
            else:
                refDataset=1
            
            if fileName in taxaExpected:
                calculateFScore=True
            else:
                calculateFScore=False
            
            postProcess(fileName, cluster_method, refDataset, calculateFScore)
            
            newFolder=processed_data_dir.replace('Processed_Data', 'T2-' + str(t2) + '-' + cluster_method + '-' + fileName.split('.')[0] + '-Data')
            os.rename(processed_data_dir,newFolder)         
            subprocess._cleanup()
            gc.collect()
