# Generate JSON data for 100 riders
import json
import os
import random

# Create a list to store rider data
riders = []
#
# # Attributes for hardworking riders
# hardworking_riders = [
#     "Friendly, hardworking",
#     "Optimistic, hardworking",
#     "Calm, hardworking",
#     "Energetic, hardworking",
#     "Perfectionist, hardworking"
# ]
#
# # Attributes for lazy riders
# lazy_riders = [
#     "Lazy, laid-back",
#     "Lazy, dependable",
#     "Lazy, charming",
#     "Lazy, practical",
#     "Lazy, efficient"
# ]
names = ["Alex Chen", "Mia Torres", "Ethan Patel", "Sophia Lee", "Olivia Smith",
         "Liam Brown", "Noah Wilson", "Ava Johnson", "James Carter", "Emma Davis",
         "Daniel Martinez", "Lucy Harris", "Michael Nguyen", "Isabella Rivera", "Jack Taylor",
         "Emily Clark", "Henry Moore", "Grace Lopez", "David Hall", "Chloe Lewis",
         "Ella King", "Ryan Young", "Sarah Adams", "Kevin Baker", "Anna Scott",
         "Jason Hill", "Lauren Allen", "Matthew Green", "Hannah Parker", "Andrew Murphy"]
genders = ["Male", "Female"]
backgrounds = [
    "a former retail worker seeking a flexible job",
    "a student working part-time to cover expenses",
    "a parent balancing work and family responsibilities",
    "a former office worker who enjoys being on the move",
    "an outdoor enthusiast looking for an active job",
    "a recent high school graduate exploring career options",
    "an experienced driver seeking a steady income",
    "a fitness enthusiast using delivery as a way to stay active",
    "a freelancer looking for supplementary income",
    "an individual transitioning from a different industry"
]
hardworking_rider_num = 34
lazy_rider_num = 33
competitive_rider = 33
for i in range(hardworking_rider_num):
    name = random.choice(names)
    age = random.randint(20, 50)
    gender = random.choice(genders)
    background = random.choice(backgrounds)
    rider = {
        "role_description": f"You are {name}, a {age}-year-old {gender} delivery rider who is {background}.",
        "personality": 'hard_working,You are a hard-working rider who will work longer hours and take more orders to earn more money.'
    }
    riders.append(rider)

for i in range(lazy_rider_num):
    name = random.choice(names)
    age = random.randint(20, 50)
    gender = random.choice(genders)
    background = random.choice(backgrounds)
    rider = {
        "role_description": f"You are {name}, a {age}-year-old {gender} delivery rider who is {background}.",
        "personality": 'lazy,You are a lazy rider, you shorten your working hours and take fewer orders because of your laziness'
    }
    riders.append(rider)

for i in range(competitive_rider):
    name = random.choice(names)
    age = random.randint(20, 50)
    gender = random.choice(genders)
    background = random.choice(backgrounds)
    rider = {
        "role_description": f"You are {name}, a {age}-year-old {gender} delivery rider who is {background}."
                            f"",
        "personality": 'competitive,You are a competitive rider. You will choose to work longer hours and take more orders because you are ranked low.'
    }
    riders.append(rider)

file_path = "rider_config_all.json"
with open(file_path, "w", encoding='utf-8') as file:
    json.dump(riders, file, indent=4)
print(f"JSON 文件已成功保存到: {file_path}")
