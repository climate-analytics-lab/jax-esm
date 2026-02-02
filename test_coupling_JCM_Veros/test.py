from acc import ACCSetup


veros_model = ACCSetup()
veros_model.setup()

print("Type of variables...")
print(type(veros_model.state.variables.salt))
