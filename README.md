# Simple Text Extractor
This program performs intelligent character recognition on images in a fast and simple way by extracting and classifying individual letters. Character model is trained on a custom CNN in PyTorch using the EMNIST Dataset.

## Sample
<div align="center">
 <img src='https://github.com/mohitpg/icr/blob/main/test_img2.jpg?raw=true'>
</div>

Result
```  
THE QUICK BROWN FOX
JUMPED OVER A LAZY DOG
```
<div align="center">
 <img src='https://github.com/mohitpg/icr/blob/main/test_img1.jpg?raw=true'>
</div>

 Result
```
MY NUMBER I5 88XXX03034
```

## Working
Replace test_img2 by the required image or change the path in the line ```img = cv2.imread('./test_img2.jpg')``` in textExtractor.py
