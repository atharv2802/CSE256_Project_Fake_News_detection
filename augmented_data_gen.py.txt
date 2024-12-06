import openai
import random
import pandas as pd


openai.api_key = "api_key" ## I used my api key

# Function to generate a single article
def generate_article(prompt, max_tokens=150):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"Error generating article: {e}")
        return None

# Prompts for real and fake sports news
real_sports_prompt = "Write a brief real sports news article about a recent game or event."
fake_sports_prompt = "Write a brief fake sports news article about an imaginary game or event."

# Generate samples
def generate_samples(num_real, num_fake):
    real_articles = [generate_article(real_sports_prompt) for _ in range(num_real)]
    fake_articles = [generate_article(fake_sports_prompt) for _ in range(num_fake)]
    
    return real_articles, fake_articles

# Generate 2000 real and 1500 fake articles
num_real = 2000
num_fake = 1500
print("Generating real articles...")
real_articles, fake_articles = generate_samples(num_real, num_fake)

# Save real articles to a CSV file
real_articles_df = pd.DataFrame({'Article': real_articles})
real_articles_df.to_csv('real_articles_augmented.csv', index=False)

# Save fake articles to a CSV file
fake_articles_df = pd.DataFrame({'Article': fake_articles})
fake_articles_df.to_csv('fake_articles_augmented.csv', index=False)

print("Real and fake articles have been saved to 'real_articles.csv' and 'fake_articles.csv'.")