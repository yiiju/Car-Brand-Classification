import os

# Create model path directory
if not os.path.exists('./modelPath/'):
    os.mkdir('./modelPath/')
     print("Directory ", './modelPath/',  " Created")

# Create result inages directory
if not os.path.exists('./resultImg/'):
    os.mkdir('./resultImg/')
     print("Directory ", './resultImg/',  " Created")

# Create test result directory
if not os.path.exists('./testResult/'):
    os.mkdir('./testResult/')
     print("Directory ", './testResult/',  " Created")
