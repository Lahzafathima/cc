dict1 = {}
key =  input("enter the roll no.: ")
value = input("enter the name: ")
dict1[key] = value
print(dict1)

while True:
    key = input("enter roll no.('or done when finish): ")
    if key.lower() == 'done':
        break
    value = input("Enter name: ")
    dict1[key] = value
print("updated dict:", dict1)

data_entries = int(input("how many key-value pairs do youwant to add? "))
for i in range(data_entries):
    key = input("enter roll no.: ")
    value = input("enter name: ")
    dict1[key] = value
print("updated dictionary: ", dict1)
print("before change directory: ", dict1)

key_change = input("enter roll no. whose name you want to change: ")
if key_change in dict1:
    new_value = input(f"enter the new value for '{key_change}': ")
    dict1[key_change] = new_value
    print("dictionary after change: ", dict1)
else:
    print(f"error: key '{key_change}' not found in dictionary.")

key_remove = input("enter key to remove: ")
try:
    del dict1[key_remove]
    print(f"key '{key_remove}' removed successfully. new dictionary: {dict1}")
except KeyError:
    print(f"error '{key_remove}' not found in dictionary")

key_search = input("enter name for search: ")
if key_search in dict1:
    print(f"key '{key_search}' is found")