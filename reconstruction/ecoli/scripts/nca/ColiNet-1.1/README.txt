This is version 1.1 of the transcriptional interactions database.

This is an updated data set of the data used for the Nature Genetics paper titled:
"Network motifs in the transcriptional regulation network of Escherichia coli"
S S SHEN-ORR, R MILO, S MANGAN & U ALON, Nature Genetics Volume 31 No. 1 pp 64 - 68.

The files in this directory include:

regInterFullFiltered.html - A listing of all the interactions in the data in an HTML table.
 
coliInterFullVec.txt - All interactions in the dataset listed as a three column vector (operon number, transcription factor number, regulation type (1 activator, 2 repressor, 3 dual)).

coliInterNoAutoRegVec.txt - A binary (unsigned) version of coliInterFullVec only without autoregulation interactions.

coliInterFullNames.txt - A two column vector, the first column is a number, matching the number of the first two columns of the above interaction files, and the second column is the name of the operon.

RegulonDB data is of version 3.2 XML version with the unknown interactions, and the chip only interactions (00001) NOT included.
All other data collected by Shai Shen-Orr under the supervision of Dr. Uri Alon

There are 578 interactions in the matrix (519 with no autoregulation).

There are three corrections done from version 1.0:

1) moaABDE is actually spelled maoABDE
2) hycA should is a part of the operon hycABCDE and therefore should be united and autoregulated.
3) ArcA -> sdh-bXXXX-something is missing.
