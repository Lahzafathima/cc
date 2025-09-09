cart=["candy", "noodles", "chips","juice","corn", "chicku"]
print(cart)
cart.append("spicy fries")
print(cart)
cart.remove("chicku")
print(cart)
cart[3]="lemo"
print(cart)

for index, item in enumerate(cart):
    print(index, item)
    
if"chips" in cart:
    print("chips found at index:", cart.index("chips"))
      
        
print("Sliced cart(1:4):", cart[1:4])

cart.sort(reverse=True)
print(cart)

length = len(cart)
print(length)
cart.remove("lemo")
print(cart)