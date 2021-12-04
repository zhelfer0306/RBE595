import os

directory = "C:\\Users\\zhelf\\OneDrive\\Desktop\\Grad School\\RBE595\\Week 12\\HW 7-8\\CroppedYale\\CroppedYale"
print("Working in: ", directory)
for dir in os.listdir(directory):
    print(dir)
    for file in os.listdir(directory+"//"+dir):
        file = str(file)
        print(file)
        # Prune out files
        if "Ambient" in file:
            print("Removing Ambient Image...")
            os.remove(directory+"//"+dir+"//"+file)
        if (file.endswith(".info") or file.endswith(".LOG")):
            print("Removing Metadata File...")
            os.remove(directory+"//"+dir+"//"+file)



