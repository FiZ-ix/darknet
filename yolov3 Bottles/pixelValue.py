from PIL import Image

im = Image.open('SpecialDark12Pack00005.jpg','r')

pix_val = list(im.getdata())
print(pix_val)

