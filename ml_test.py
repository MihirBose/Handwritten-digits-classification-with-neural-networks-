import pickle

# Best params
infile = open('nnscript_params/params','rb')
#infile = open('C:/Users/mihir/Desktop/params','rb')

obj1 = pickle.load(infile)
obj2 = pickle.load(infile)
obj3 = pickle.load(infile)
obj4 = pickle.load(infile)
obj5 = pickle.load(infile)
print (obj1, obj2, obj3, obj4, obj5)
print (obj3.shape)
print (obj4.shape)
infile.close()

# # All params
# infile = open('nnscript_params/all_hyperparams','rb')
# obj1 = pickle.load(infile)
# for item in obj1:
#     print (item[2:6]) # Not printing the weights here(0 and 1 index in item)
#     #print ('\n')
# infile.close()

#print (new_dict)