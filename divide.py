import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

current_dir = '/Users/isabellayu/darknet/data/hoops/images'

# Percentage of images to be used for the test set
percentage_test = 20

# Create and/or truncate train.txt and test.txt
file_train = open('cfg/train.txt', 'w')
file_test = open('cfg/test.txt', 'w')  # Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in os.listdir(current_dir):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    print(title)
    if counter == index_test:
        counter = 1
        file_test.write(current_dir + title + '.jpg' + "\n")
    else:
        file_train.write(current_dir + title + '.jpg' + "\n")
        counter = counter + 1