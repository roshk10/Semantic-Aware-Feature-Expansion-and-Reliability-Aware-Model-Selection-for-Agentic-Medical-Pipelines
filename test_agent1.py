from agents.agent1_intake import run

# Test 1: valid CSV
result = run("data/sample.csv")
print(result["status"])
print(result["columns"])
print(result["pii_flag"])

# Test 2: missing file
result = run("data/doesnt_exist.csv")
print(result["status"])
print(result["error_message"])