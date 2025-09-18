from PIL import Image
img = Image.open("/home/mila/o/ozgur.aslan/git/LIBERO/libero/libero/assets/textures/table_blue_plastic.png")
print(img.size)  # (width, height)

cropped = img.crop((0, 0, 1024, 1024))
cropped.save("/home/mila/o/ozgur.aslan/git/LIBERO/libero/libero/assets/textures/table_blue_plastic.png")