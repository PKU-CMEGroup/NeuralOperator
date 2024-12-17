# Data Information

Download ahmed body  data from 

PKU drive

https://disk.pku.edu.cn/link/ARF1718772A25348CCB4FD34073F571444

Name: ahmed_body.zip

File Location: AnyShare://Neural-Operator-Data/ahmed_body.zip


# Features
The features variable has the shape (ndata, nnodes, 9), which consists of 9 dimensions, where each dimension represents specific information:

Output
- 0 mean pressure of neighbour triangles

Infos
- 1 length
- 2 width
- 3 height
- 4 clearance
- 5 slant
- 6 radius
- 7 velocity
- 8 reynolds number
  
Infos are normalized to the range $[0,1]$, which is necessary to ensure stable and efficient training. 