import os
from dotenv import load_dotenv
from openai import OpenAI
import re
import requests
import shutil

# load .env file
env = os.getenv("ENVIRONMENT", "development")
dotenv_path = f".env.{env}"
load_dotenv(dotenv_path=dotenv_path)


def get_ingredients():
    """Get user ingredients and insert into a list.

    Returns:
        - items (list) - a list of ingredients.
    """
    items = []

    while True:
        user_input = input("Enter items separated by comma (type 'done' to continue): ")

        if user_input.lower() == "done":
            break

        items.extend(item.strip() for item in user_input.split(","))

    return items


def create_dish_prompt(list_of_ingredients):
    """Structures and formats the prompt.

    Args:
        - list_of_ingredients (list)
    Returns:
        - prompt (string) - a structured and correctly
          formatted prompt.
    """
    prompt = (
        f"Create a detailed recipe based on only the following ingredients: {', '.join(list_of_ingredients)}.\n"
        + f"Additionally assign a title starting with 'Recipe Title: ' to this recipe."
    )
    return prompt


def extract_title(recipe):
    """Extract the title elements from the prompt.

    Args:
        - recipe ingredients
    Returns:
        - a structured, formatted prompt for image title
    """
    return (
        re.findall("^.*Recipe Title: .*$", recipe, re.MULTILINE)[0]
        .strip()
        .split("Recipe Title: ")[-1]
    )


def dalle3_prompt(recipe_title):
    """Generate, structure and format the image prompt.

    Args:
        - recipe title
    Returns:
        - prompt
    """
    prompt = f"{recipe_title}, professional food photography, 15mm, studio lighting"
    return prompt


def save_image(image_url, file_name):
    """Saves the image generated.

    Args:
        - image_url
        - file_name
    Returns:
        - image status code (200 OK hopefully)
    """
    image_res = requests.get(image_url, stream=True)
    if image_res.status_code == 200:
        with open(file_name, "wb") as f:
            shutil.copyfileobj(image_res.raw, f)
    else:
        print("ERROR LOADING IMAGE")

    return image_res.status_code


# start client
client = OpenAI()

# retrieve api key
client.api_key = os.getenv("OPENAI_API_KEY")

list_of_ingredients = get_ingredients()

recipe = create_dish_prompt(list_of_ingredients)

# submit the response
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": recipe,
        },
    ],
    temperature=0.7,
    max_tokens=512,
)

# print the response to the screen
print(response.choices[0].message.content, end="\n\n")

# store the result text
result_text = response.choices[0].message.content


# extract title from the result text
recipe_title = extract_title(result_text)


# Generate Dish Image
response = client.images.generate(
    model="dall-e-3",
    prompt=dalle3_prompt(recipe_title),
    size="1024x1024",
    quality="hd",
    n=1,
)

# store the image url
image_url = response.data[0].url
print(image_url)


# Save the image
save_image(image_url, "example_download.png")
