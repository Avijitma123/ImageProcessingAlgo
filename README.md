# Soft-set MSER

#I have determined the optimal threshold value using the Softset concept, resulting in Maximally Stable Extremal Regions (MSER) for the given input image at the threshold value of 138. It's noteworthy that as I increment the threshold value beyond 138 to 139, the extremal regions persist in stability, indicating a robust representation under the adjusted threshold.


Steps to follow to complete the thresholding process is—<br><br>
               STEP 1: At first I convert the grayscale image into binary image.<br>
               STEP 2: Next, I have pass the binary image into connectedComponentsWithStats() function. The connected component function will return ------<r>
               
1)	IDs of all unique connected components of the input image.
2)	The starting x coordinate of the component.
3)	The starting y coordinate of the component.
4)	The width (w) of the component.
5)	The height (h) of the component.
6)	Return a Labels. A mask named labels have the same spatial dimensions as our input thresh image. For each location in labels, we have an integer ID value that corresponds to the connected component where the pixel belongs.<br>
STEP 3: Using the Labels matrix, We get a set of pixel coordinates that belong to the corresponding component (ID). <br>
STEP 4: Now I track the component for different threshold value. 
                   Let, for threshold t1, the Soft set (α, t1)   = {(x1, y1), (x2, y2), ……., (xn, yn)}  
                                 and for threshold t2 , the Soft set  (α,t2)   ={ (x1,y1), (x2,y2)…….(xn,yn) }  <br>
STEP 5:    We take the Soft Set Difference function at different thresholds. Then we find cardinality of the difference set. Whichever threshold gives lowest cardinality is the threshold where it is maximally stable.
D α   =(α,t1) - (α,t2)      [Here I have calculated the normal set difference operation using the list in Python] <br>
STEP 6: Nex we calculate the cardinality of D α .
     𝛥1α= | D α |= |(𝑡1,α )− (𝑡2,α )|

we will repeat these steps for all components of the image ,for all threshold value  100 <= t <=220.
We find 𝛥1α. Similarly, for each threshold we can find 𝛥2α, 𝛥3α…and we store these values into a list. Then we have to find which threshold gives minimum 𝛥 to find maximally stable region.
       T=min( [𝛥1α,………. 𝛥nα]) [ T will be the threshold , gives minimum 𝛥 to find maximally stable region]
