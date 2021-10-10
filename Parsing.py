import os
import pickle
from bs4 import BeautifulSoup
from PIL import Image

# Create a Features directory
feature_dir = 'D:\Datasets\Features'
if os.path.isdir(feature_dir) == False:
    print('Creating "Features" directory', feature_dir)
    os.mkdir(feature_dir)
    if os.path.isdir(feature_dir) == True:
        print('Features directory created')
    else: 
        print('Error creating features directory')
else:
    print('Features directory already exists')

# Iterate through Annotations directory (this directory is extracted from the downloaded tarball)
annotation_dir = 'D:\Datasets\Annotations'
jpg_dir = 'D:\Datasets\JPEGImages'
for entry in os.scandir(annotation_dir):
    with open(entry.path, 'r') as xml:
        data = xml.read()
    # Parse each xml file in the Annotations directory using Beautifulsoup
    xml_data = BeautifulSoup(data, "xml")
    for nametag in xml_data.select('filename'):
        print(nametag.get_text())
        # Create a subdirectory inside the Features directory for storing this image's features
        imagename_dir = feature_dir+'\\'+nametag.get_text()
        os.mkdir(imagename_dir)
        # Create a subdirectory for each one of this image's objects
        object_count = 0
        for object in xml_data.find_all('object'):
            object_dir = imagename_dir+'\\'+object.find('name').get_text()
            if os.path.isdir(object_dir) == False:
                os.mkdir(object_dir)
            else:
                object_count += 1
                object_dir = object_dir+str(object_count)
                os.mkdir(object_dir)
            # Place a cropped image of the object in its directory
            for point in object.find_all('bndbox'):
                left = int(point.find('xmin').get_text())
                right = int(point.find('xmax').get_text())
                top = int(point.find('ymax').get_text())
                bottom = int(point.find('ymin').get_text())
            im = Image.open(jpg_dir+'\\'+nametag.get_text())
            im_crop = im.crop((left, bottom, right, top))
            """
            im_crop = im_crop.resize((224,224), resample = Image.NEAREST)    # Uncomment to resize the image to 224x224. Even with a resampler, it's not ideal.
            """
            im_crop.save(object_dir+'\\'+object.find('name').get_text()+'_Image.jpg', 'JPEG')
            # Pickle the object image to the binary database
            with open('D:\Datasets\Features\PASCAL_db.bin', 'wb') as f:
                pickle.dump(im_crop, f, pickle.HIGHEST_PROTOCOL)
            # Create a subdirectory for each object's parts
            part_count = 0
            for part in object.find_all('part'):
                part_dir = object_dir+'\\'+part.find('name').get_text()
                if os.path.isdir(part_dir) == False:
                    os.mkdir(part_dir)
                else:
                    part_count += 1
                    part_dir = part_dir+str(part_count)
                    os.mkdir(part_dir)
                # Place a cropped image of the part in its directory
                for point in part.find_all('bndbox'):
                    left = int(part.find('xmin').get_text())
                    right = int(part.find('xmax').get_text())
                    top = int(part.find('ymax').get_text())
                    bottom = int(part.find('ymin').get_text())
                im = Image.open(jpg_dir+'\\'+nametag.get_text())
                im_crop = im.crop((left, bottom, right, top))
                im_crop.save(part_dir+'\\'+part.find('name').get_text()+'_Image.jpg', 'JPEG')
                # Pickle the part image into the binary database
                with open('D:\Datasets\Features\PASCAL_db.bin', 'wb') as f:
                    pickle.dump(im_crop, f, pickle.HIGHEST_PROTOCOL)




