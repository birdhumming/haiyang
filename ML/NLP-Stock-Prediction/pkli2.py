import csv
import pickle


class primaryDetails:
    def __init__(self, name, age, gender, contactDetails):
        self.name = name
        self.age = age
        self.gender = gender
        self.contactDetails = contactDetails

    def __str__(self):
        return "{} {} {} {}".format(self.name, self.age, self.gender, self.contactDetails)

    def __iter__(self):
        return iter([self.name, self.age, self.gender, self.contactDetails])

class contactDetails:
    def __init__(self, cellNum, phNum, Location):
        self.cellNum = cellNum 
        self.phNum = phNum
        self.Location = Location

    def __str__(self):
        return "{} {} {}".format(self.cellNum, self.phNum, self.Location)

    def __iter__(self):
        return iter([self.cellNum, self.phNum, self.Location])


a_list = []

with open("lemmatized_text.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        a = contactDetails(row[3], row[4], row[5])
        a_list.append(primaryDetails(row[0], row[1], row[2] , a))

file = open('lemmatized_text.pkl', 'wb')
# pickle.dump(a_list[0], primaryDetails)
#pickle.dump(primaryDetails, a_list[0])
with open('writepkl.pkl', 'wb') as output_file:
    pickle.dump(a_list, output_file)

file.close()
