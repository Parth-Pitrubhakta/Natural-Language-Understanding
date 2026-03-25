"""
Script to add common Indian last names to the existing first names in TrainingNames.txt.
Creates full names (First Last) format.
"""
import random

# Common Indian last names across different regions
LAST_NAMES = [
    # North Indian
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Agarwal", "Jain", "Mehta",
    "Chauhan", "Tiwari", "Pandey", "Mishra", "Dubey", "Srivastava", "Saxena",
    "Kapoor", "Khanna", "Malhotra", "Arora", "Bhatia", "Chopra", "Dhawan",
    "Goel", "Joshi", "Kashyap", "Mathur", "Nigam", "Rastogi", "Tandon",
    # South Indian
    "Nair", "Menon", "Iyer", "Iyengar", "Reddy", "Rao", "Naidu", "Pillai",
    "Krishnan", "Subramaniam", "Venkatesh", "Rajan", "Anand", "Bhat",
    "Hegde", "Shetty", "Pai", "Kamath", "Prabhu", "Kulkarni",
    # West Indian
    "Patel", "Desai", "Shah", "Parekh", "Joshi", "Bhatt", "Trivedi",
    "Doshi", "Naik", "Patil", "Deshpande", "Gokhale", "Karnik",
    # East Indian
    "Banerjee", "Chatterjee", "Mukherjee", "Das", "Bose", "Sen",
    "Ghosh", "Roy", "Dutta", "Sarkar", "Chakraborty", "Ganguly",
    # Sikh
    "Gill", "Dhillon", "Sidhu", "Bajwa", "Sandhu", "Randhawa", "Brar",
    # General
    "Sethi", "Mehra", "Ahuja", "Soni", "Thakur", "Chandra", "Prasad",
    "Mohan", "Rathore", "Chahar", "Yadav", "Rawat", "Bisht", "Negi"
]

def main():
    random.seed(42)

    # Read existing first names
    with open("TrainingNames.txt", "r") as f:
        first_names = [line.strip() for line in f if line.strip()]

    # Assign a random last name to each first name
    full_names = []
    for first in first_names:
        last = random.choice(LAST_NAMES)
        full_names.append(f"{first} {last}")

    # Write back
    with open("TrainingNames.txt", "w") as f:
        f.write("\n".join(full_names) + "\n")

    print(f"Updated {len(full_names)} names with last names.")
    print(f"Sample: {full_names[:5]}")


if __name__ == "__main__":
    main()
