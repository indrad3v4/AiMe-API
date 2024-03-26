import requests
import logging
from openai import OpenAI
import os
import sys
from bs4 import BeautifulSoup
from roles import strategy_roles_dict, creative_roles_dict, producing_roles_dict, media_roles_dict
import re
import json
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from model import nlp
from werkzeug.utils import secure_filename
from spacy.matcher import PhraseMatcher
from textblob import TextBlob
from PIL import Image
import io
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage


from collections import OrderedDict

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Path to your service account JSON file
cred_path = 'aime.json'

# Initialize the app with a service account, granting admin privileges
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'aime-5b083.appspot.com'
})

class Controller:
    def __init__(self, agency, view):
        self.agency = agency
        self.view = view
        self.client_data = {}
        self.concatenated_responses = []
        self.shopping_list = []
        self.history_stack = []  # To keep track of user's path
    
    
    def list_images_in_static(self):
        static_path = './static'  # Assuming 'static' directory is in the current working directory
        try:
            # List all image files in the static directory
            image_files = [f for f in os.listdir(static_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            return image_files
        except FileNotFoundError:
            self.view.display_message(f"Static directory not found at path: {static_path}.", style="error")
            return None
        except Exception as e:
            logging.error(f"An error occurred while listing images: {e}")
            self.view.display_message("An error occurred while listing images.", style="error")
            return None
    
    
    def collect_initial_client_data(self, form_data, image_file=None):
        # Resetting or clearing the image data
        self.client_data['image_base64'] = None

        # Define all the required form fields
        required_fields = ['brand_name', 'product_service', 'goals', 'target_audience', 
                           'audience_insight', 'message', 'city', 'time', 'budget', 'big_idea', 'channels']

        # Validate form_data to contain all required fields
        for field in required_fields:
            if field not in form_data or not form_data[field].strip():
                raise ValueError(f"The field '{field}' is missing or empty in the form data.")

        # Directly assign validated form data to client_data
        self.client_data.update({field: form_data[field] for field in required_fields})

        # Handle image uploading and encoding to base64
        if image_file:
            # Save the image and get its path
            image_path = self.save_image(image_file)
            if image_path:
                # Encode the saved image to base64
                self.client_data['image_base64'] = self.encode_image_to_base64(image_path)
            else:
                logging.error("Image was not saved successfully.")

        # Process and tailor the client's data
        narrative = self.create_client_narrative(self.client_data)
        doc = nlp(narrative)  # Process the narrative with the spaCy model
        themes, sentiment = self.extract_themes_and_sentiment(doc)

        # Generate a tailored response based on the processed data
        tailored_response = self.generate_tailored_response(self.client_data)

        # Return the filled client_data dictionary
        return self.client_data


    def save_image(self, image_file):
        if image_file:
            filename = secure_filename(image_file.filename)
            # Make sure the 'static/images' directory exists
            if not os.path.exists('static/images'):
                os.makedirs('static/images')
            image_path = os.path.join('static/images', filename)
            image_file.save(image_path)
            return image_path
        return None

    
    def encode_image_to_base64(self, image_path, max_size=512):
        try:
            with Image.open(image_path) as img:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')  # Use a common format for the web
                encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return encoded_string
        except Exception as e:
            logging.error(f"Error encoding image to base64: {e}")
            return None
    
    
    
    def create_client_narrative(self, client_data):
        narrative = []
        for key, value in client_data.items():
            if value:  # Ensure the value is not None or empty
                narrative.append(f"{key.replace('_', ' ').capitalize()}: {value}")
        return ". ".join(narrative) + "."
    
    
    def extract_themes_and_sentiment(self, narrative):
        doc = nlp(narrative)  # Ensure narrative is processed by spaCy to create a Doc object
        matcher = PhraseMatcher(nlp.vocab)
        patterns = [nlp.make_doc(text) for text in ["brand", "campaign", "audience", "strategy", "engagement"]]
        matcher.add("THEME", patterns)
        themes = [doc[start:end].text for _, start, end in matcher(doc)]  # Use the doc here
        sentiment = TextBlob(doc.text).sentiment.polarity
        return themes, sentiment
    
    
    def generate_tailored_response(self, client_data):
        # Create a narrative from the client data
        narrative = self.create_client_narrative(client_data)

        # Process the narrative with spaCy to get a document object
        doc = nlp(narrative)

        # Extract themes and sentiment from the document
        themes, sentiment = self.extract_themes_and_sentiment(doc)

        # Generate insights from dependency parsing
        insights = self.generate_insights_from_dependency_parsing(doc)

        # Describe the overall sentiment based on the analysis
        sentiment_description = 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'

        # Assuming 'channels' is provided in client_data and is a comma-separated string of channels
        channels = client_data.get('channels', '').split(',')

        # Initialize a dictionary to hold strategies for each channel
        channel_strategies = {}

        # Loop over each channel to generate tailored responses
        for channel in channels:
            channel_trimmed = channel.strip()
            channel_prompt = (
                         f"Considering the themes {', '.join(themes)} with an overall sentiment of {sentiment_description}, "
                         f"and the insight: {client_data.get('audience_insight', 'No audience insight provided')}, "
                         f"craft a tailored messaging strategy for the {channel_trimmed} channel. "
                         f"\n\nClient's input: {narrative}\n\n"
                         f"Insights from dependency parsing:\n{insights}"
                     )

                     # Use the OpenAI API or another method to generate the response for this channel
            channel_response = self.call_openai_api(channel_prompt)

            # Store the response in the channel_strategies dictionary
            channel_strategies[channel_trimmed] = channel_response

        return channel_strategies

    
    
    def call_openai_api(self, prompt, insights=None, image_base64=None, is_dalle=False, client_data=None, image_file=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }

        # Initialize client_data to an empty dictionary if it's None
        if client_data is None:
            client_data = {}

        if image_file:
            # Assuming save_image() is a method that saves the image and returns the path
            image_path = self.save_image(image_file)
            # Use a predefined max size or define it appropriately
            max_size = 512
            base64_image = self.encode_image_to_base64(image_path, max_size)
        elif 'image_base64' in client_data and client_data['image_base64']:
            # If image_base64 is already in client_data, use that
            base64_image = client_data['image_base64']
        else:
            base64_image = None
    
        # Incorporate insights into the prompt if they are provided
        if insights:
            prompt += f"\n\nInsights from dependency parsing:\n{insights}"
    
        try:
            # For generating an image using DALL-E
            if is_dalle and client_data:
                tailored_response = self.generate_tailored_response(client_data)
                themes, sentiment = self.extract_themes_and_sentiment(tailored_response)
                brand_name = client_data.get('brand_name', 'the brand')
                audience_insight = client_data.get('audience_insight', 'Audience Insight')
    
                dalle_prompt = (
                    f"Act as an advertising design expert to create a high-quality advertising image that encapsulates '{brand_name}', "
                    f"resonating with the sentiment '{sentiment}' and emphasizing the brand's vision. "
                    f"Accentuate themes: {', '.join(themes)}. "
                    f"Visualize the brand in a context appealing to {audience_insight}, "
                    f"and incorporate a color palette that aligns with the brand's values."
                )
    
                response = requests.post(
                    "https://api.openai.com/v1/images/generations",
                    headers=headers,
                    json={
                        "prompt": dalle_prompt,
                        "n": 1,
                        "size": "1792x1024",
                        "model": "dall-e-3",  # Specify the model version
                        "quality": "hd",  # Specify the quality of the image
                        "style": "vivid"  # Specify the style of the image
                    }
                )
    
                if response.status_code == 200:
                    image_data = response.json().get('data', [{}])[0]
                    image_url = image_data.get('url')
                    logging.info(f"Generated image URL: {image_url}")
                    return image_url
                else:
                    logging.error("DALL-E API call did not return any data.")
                    return "Unable to generate image."
    
            # For analyzing or generating text based on a base64 encoded image
            elif image_base64:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "data": image_base64}
                    ]
                }]
                model = "gpt-4-vision-preview"
                max_tokens = 512
    
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                )
    
                if response.status_code == 200:
                    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                else:
                    logging.error(f"API call failed with status code {response.status_code}")
                    return "Unable to generate response."
    
            # Default text-based API call without an image
            else:
                messages = [{"role": "system", "content": "This is a message from the system."},
                            {"role": "user", "content": prompt}]
                model = "gpt-4-0125-preview"
                max_tokens = 1000
    
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                )
    
                if response.status_code == 200:
                    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                else:
                    logging.error(f"API call failed with status code {response.status_code}")
                    return "Unable to generate response."
    
        except Exception as e:
            logging.error(f"An error occurred during the API call: {e}")
            return f"An error occurred while generating the response: {e}"
    
    
    
    def run(self):
        logging.debug("Starting the run method.")
    
        # Loop to allow for multiple campaign briefings
        while True:
            # Clear the client data and other states at the beginning of each campaign briefing
            self.client_data.clear()
            self.concatenated_responses = []
            self.shopping_list = []
            self.history_stack = []
    
            self.view.display_message(f"Welcome to {self.agency.name} Ad Agency! Let's craft an impactful advertising campaign.")
    
            # Collect initial client data
            self.client_data = self.collect_initial_client_data()
    
            # Interactive loop for departmental work or finalizing campaign
            campaign_finalized = False
            while not campaign_finalized:
                self.view.display_message("\nWhat would you like to do next? You can choose a department to work with, go back to the previous step, or finalize your campaign.")
    
                # Show options
                for idx, dept_name in enumerate(self.agency.departments.keys(), 1):
                    self.view.display_message(f"{idx}. Work with {dept_name} Department")
                self.view.display_message("0. Finalize Campaign and Send Presentation")
    
                # Get user choice
                choice = self.view.get_user_input("Please enter your choice or type 'exit' to end the session: ")
    
                # Handle choice
                if choice.lower() == 'back':
                    self.backtrack()
                elif choice.lower() == 'exit':
                    self.view.display_message("Ending session. Thank you for using the A&I Ad Agency software!")
                    return  # Exit the method
                elif choice.isdigit() and int(choice) == 0:
                    final_presentation = self.finalize_campaign()
                    if final_presentation:
                        campaign_finalized = True
                        self.view.display_message("Campaign finalized successfully.", style="success")
                    else:
                        self.view.display_message("An error occurred during campaign finalization.", style="error")
                else:
                    self.handle_department_choice(choice)
    
            # Save the session results after finalizing
            if final_presentation:
                save_success = self.save_session_results(self.client_data)
                if not save_success:
                    self.view.display_message("Failed to save session results.", style="error")
    
                # Email the final presentation if requested
                if self.view.confirm_action("Would you like this presentation emailed to you?"):
                    email = self.view.get_user_input("Please enter your email address:")
                    if self.validate_email(email):
                        self.send_final_presentation(final_presentation, email)
                        self.view.display_message(f"Email sent successfully to {email}.")
                    else:
                        self.view.display_message(f"The email address entered appears to be invalid: {email}")
    
    
            # Prompt to start a new campaign at the end of the current campaign
            if not self.view.confirm_action("Would you like to start a new campaign?"):
                self.view.display_message("Thank you for choosing the AiMe Ad Agency software! Want more? Contact us.")
                break  # Exit the while loop and end the run method
    
    
    
    
    
    def finalize_campaign(self, client_data):
        # Generate the client narrative and process it with spaCy
        client_narrative = self.create_client_narrative(client_data)
        doc = nlp(client_narrative)

        # Extract themes and sentiment from the narrative
        themes, sentiment = self.extract_themes_and_sentiment(doc)

        # Generate a tailored response based on themes and sentiment
        tailored_response = self.generate_tailored_response(client_data)

        # Call the OpenAI API to generate unique text for different sections of the final presentation
        big_idea_prompt = "Instruction for you chatgpt is next: Act as TRIZ-Creator for Generate a 'big idea' that encapsulates drama, form, and value behind a campaign for {product_service} targeting {target_audience}.".format(
            product_service=client_data.get('product_service', 'the product'),
            target_audience=client_data.get('target_audience', 'the target audience')
        )
        big_idea = self.call_openai_api(big_idea_prompt)

        activation_mechanics = self.call_openai_api("Generate text for activation mechanics based on strategic milestones for {brand_name}.".format(brand_name=client_data.get('brand_name', 'the brand')))
        production_plan = self.call_openai_api("Generate text for production plan based on campaign content creation for {brand_name}.".format(brand_name=client_data.get('brand_name', 'the brand')))
        media_plan = self.call_openai_api("Generate text for media plan and KPI funnel based on success benchmarks and audience engagement tactics for {brand_name}.".format(brand_name=client_data.get('brand_name', 'the brand')))
        conclusion = self.call_openai_api("Generate text for conclusion and next steps based on campaign belief and future developments for {brand_name}.".format(brand_name=client_data.get('brand_name', 'the brand')))

        # Concatenate insights and responses to build the final presentation
        final_presentation_content = "Final Campaign Presentation\n\n"
        final_presentation_content += f"Agency Goals and Objectives: {client_data.get('goals', 'Not provided')}\n"
        final_presentation_content += f"Strategic Approach: {tailored_response}\n"
        final_presentation_content += f"The Big Idea: {big_idea}\n"
        final_presentation_content += f"Activation Mechanics System: {activation_mechanics}\n"
        final_presentation_content += f"Production Plan: {production_plan}\n"
        final_presentation_content += f"Media Plan and KPI Funnel: {media_plan}\n"
        final_presentation_content += f"Conclusion and Next Steps: {conclusion}\n"

    
        # Generate DALL-E image if needed
        image_base64 = self.client_data.get('image_base64', None)
        if image_base64:
            # Generate the tailored response to extract themes and sentiment
            tailored_response = self.generate_tailored_response(self.client_data)
            themes, sentiment = self.extract_themes_and_sentiment(tailored_response)
            product = self.client_data.get('product', 'Product')
            audience_insight = self.client_data.get('audience_insight', 'Audience Insight')
    
            # Construct the dalle_prompt using the above variables
            dalle_prompt = (
                f"Act as voice of a&i ad_agency for Create a high-quality advertising image for advertising '{product}' in '{client_narrative}' context, "
                f"resonating with the sentiment '{sentiment}' and the brand's vision. "
                f"Emphasize themes: {', '.join(themes)}. "
                f"Show the product in a setting appealing to {audience_insight}, "
                f"highlighting unique features, style, and vibe."
            )
    
            # Pass the dalle_prompt to the API call
            dalle_image_url = self.call_openai_api(
                prompt=dalle_prompt,  # Using the generated dalle_prompt here
                insights=None,
                image_base64=image_base64,
                is_dalle=True,
                client_data=self.client_data
            )
            final_presentation_content += f"DALL-E Generated Image: {dalle_image_url}\n"
    
    
        # Append any additional content like shopping lists or department insights
        if self.concatenated_responses:
            # Convert all non-string items to strings
            concatenated_responses_str = [str(response) if not isinstance(response, str) else response for response in self.concatenated_responses]
            final_presentation_content += "Insights and Responses from Departments:\n"
            final_presentation_content += "\n".join(concatenated_responses_str) + "\n"
    
        if self.shopping_list:
            final_presentation_content += "Shopping List with Estimated Costs:\n" + \
                                          "\n".join([f"- {item}: €{cost}" for item, cost in self.shopping_list.items()]) + "\n"

        # Clear temporary data storage after use
        self.client_data.clear()
        self.concatenated_responses.clear()
        self.shopping_list.clear()
    
        # Display the final presentation
        return final_presentation_content
    
    
    
    
    def generate_insights_from_dependency_parsing(self, doc):
        insights = []
        # Define the terms that are relevant for the advertising context
        relevant_terms = ['brand', 'campaign', 'audience', 'strategy', 'engagement', 'message']
    
        # Look for these terms in the doc and gather insights related to them
        for token in doc:
            if token.text.lower() in relevant_terms:
                for child in token.children:
                    # Look for adjectives and compound nouns that modify the relevant terms
                    if child.dep_ in ("amod", "compound"):
                        insights.append(f"{child.text.capitalize()} {token.text.lower()}")
                    # Look for verbs associated with the relevant terms to understand the actions
                    elif child.dep_ == "relcl" and child.pos_ == "VERB":
                        insights.append(f"{token.text.capitalize()} that {child.text}")
    
        # Combine insights into a single string for presentation
        return "\n".join(set(insights))  # Use set to avoid duplicate insights
    
    
    def handle_final_presentation(self, final_presentation):
        # Display the final presentation
        self.view.display_presentation(Panel(final_presentation, title="Final Presentation", subtitle="Generated by A&I Ad Agency"))
    
        # Ask the user if they'd like the presentation emailed to them
        if self.view.confirm_action("Would you like emailing this presentation to you?"):
            recipient_email = self.view.get_user_input("Please enter your email address: ")
            if self.validate_email(recipient_email):
                self.send_final_presentation(final_presentation, recipient_email)
                self.view.display_message("Presentation sent successfully to your email.")
            else:
                self.view.display_message("Invalid email address. Presentation not sent.")
    
    
    def attempt_email_delivery(self, final_presentation):
        client_email = self.view.get_user_input("Please enter your email address: ")
        if self.validate_email(client_email):
            confirmation = self.view.confirm_action(f"Is this the correct email address: {client_email}?")
            if confirmation:
                self.send_final_presentation(final_presentation, client_email)
                self.view.display_message("Presentation sent successfully to your email.")
            else:
                self.view.display_message("Email sending canceled. Let's try entering your email address again.")
                self.attempt_email_delivery(final_presentation)
        else:
            self.view.display_message("The email address entered appears to be invalid. Please try again.")
            self.attempt_email_delivery(final_presentation)
    
    def validate_email(self, email):
        return re.match(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", email) is not None
    
    def handle_department_interaction(self, department_name):
      department = self.agency.get_department(department_name)
      if department:
          self.view.display_message(f"Available roles in {department.name}:")
          for idx, role in enumerate(department.roles, 1):
              self.view.display_message(f"{idx}. {role.name}")
    
          role_choice = self.view.get_user_input("Choose a role: ")
          self.handle_role_choice(department, role_choice)
      else:
          self.view.display_message(f"Department {department_name} not found.")
    
    def handle_department_choice(self, choice):
        try:
            department_index = int(choice) - 1
            department_names = list(self.agency.departments.keys())
            if 0 <= department_index < len(department_names):
                department_name = department_names[department_index]
                self.handle_department_interaction(department_name)
            else:
                self.view.display_message("Invalid choice. Please try again.")
        except ValueError:
            self.view.display_message("Please enter a valid number.")
    
    # Inside Controller class where you might be handling role selection
    def handle_role_choice(self, department, role_choice):
        try:
            role_index = int(role_choice) - 1
            if 0 <= role_index < len(department.roles):
                role = department.roles[role_index]
                # Check if the role exists in the appropriate dictionary based on the department
                if department.name == 'Strategy':
                    role_details = strategy_roles_dict.get(role.name, None)
                elif department.name == 'Creative':
                    role_details = creative_roles_dict.get(role.name, None)
                elif department.name == 'Producing':
                    role_details = producing_roles_dict.get(role.name, None)
                elif department.name == 'Media':
                    role_details = media_roles_dict.get(role.name, None)
                else:
                    role_details = None  # If the department is not recognized
    
                if role_details:
                    # If role details exist, use them to create a Role instance and interact with it
                    # This Role instance is then stored in self.role
                    self.role = Role(role.name, role_details['description'], role_details['responsibilities'], self)
                    self.interact_with_role(self.role)
                else:
                    self.view.display_message(f"Details for {role.name} not found.")
            else:
                self.view.display_message("Invalid role choice. Please try again.")
        except ValueError:
            self.view.display_message("Please enter a valid number.")
    
    def interact_with_role(self, role):
        # Format client data for role interaction
        input_data = self.view.format_client_data_for_role_interaction(self.client_data)
    
        # Retrieve client brief and last feedback (if any)
        client_brief = self.client_data.get('brief', 'No brief provided')
        last_feedback = self.client_data.get('last_feedback', 'No previous feedback')
    
        # Generate the initial response from the selected role
        response_content = role.generate_response(input_data, client_brief, last_feedback, self.client_data)
    
        # Display the role's response to the user
        self.view.display_message(f"Response from {role.name}: {response_content}")
    
        # Ask the user to rate the response
        rating = self.view.rate_response()
    
        # Handle user feedback for ratings less than 4
        if rating < 4:
            comment = self.view.get_user_comment()
            response_content += f"\nUser Comment: {comment}"
            self.client_data['last_feedback'] = comment  # Update last feedback with user comment
    
        # Append the role's response (including user comments) to the list of concatenated responses
        if not isinstance(self.concatenated_responses, list):
            self.concatenated_responses = [self.concatenated_responses]
        self.concatenated_responses.append(response_content)
    
        # Handle follow-up questions or topics from the user
        follow_up_needed = self.view.get_user_input("Do you have any follow-up questions or topics you wish to explore further? (yes/no): ")
        if follow_up_needed.lower() == 'yes':
            follow_up_question = self.view.get_user_input("Please enter your follow-up question: ")
            follow_up_response = role.generate_response(follow_up_question, client_brief, last_feedback, self.client_data)
            self.view.display_message(f"Follow-up response from {role.name}: {follow_up_response}")
            self.concatenated_responses.append(follow_up_response)
    
    
    def backtrack(self):
      if self.history_stack:
          last_department, last_role_choice = self.history_stack.pop()
          self.view.display_message(f"Backtracking to {last_department.name}, role {last_role_choice}")
          self.handle_role_choice(last_department, last_role_choice)
      else:
          self.view.display_message("No previous steps to backtrack to.")
    
    
    def handle_user_feedback(self, response):
      rating = self.view.rate_response()
      user_comment = ""
      if rating < 4:  # If the rating is below 4, ask for comments
          user_comment = self.view.get_user_comment()
      return response, user_comment
    
    
    def generate_final_presentation(self, themes, sentiment, audience_insight):
        # Start with an introduction to the presentation
        final_presentation_content = "Final Campaign Presentation\n\n"
    
        # Verify if the responses are concatenated properly
        if not self.concatenated_responses:
            self.view.display_message("No responses collected for presentation.")
            return None
    
        # Add an overview of the campaign objectives and insights
        final_presentation_content += "Campaign Overview:\n"
        final_presentation_content += f"Themes: {', '.join(themes)}\n"
        final_presentation_content += f"Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}\n"
        final_presentation_content += f"Audience Insights: {audience_insight}\n\n"
    
        # Incorporate concatenated responses into the presentation
        final_presentation_content += "Strategic Insights from Agency Roles:\n"
        for response in self.concatenated_responses:
            final_presentation_content += f"{response}\n\n"
    
        # Generate a tailored response based on the campaign overview
        tailored_response = self.generate_tailored_response(themes, sentiment, audience_insight)
        final_presentation_content += f"\nTailored Campaign Strategy:\n{tailored_response}\n"
    
        # Generate and append the shopping list
        shopping_list_items = self.role.generate_shopping_list(tailored_response) if hasattr(self.role, 'generate_shopping_list') else []
        shopping_list = "\n".join([f"- {item}" for item in shopping_list_items])
        final_presentation_content += f"\nRequired Resources & Estimated Costs:\n{shopping_list}\n"
    
        # Generate image with DALL-E
        product_image_url = self.call_openai_api(prompt="Generate a creative image for the product.", is_dalle=True, client_data=self.client_data)
        if "http" in product_image_url:  # Simple check to see if it's likely a URL
            final_presentation_content += f"\nProduct Image Visualization:\n{product_image_url}\n"
        else:
            final_presentation_content += "\nProduct Image Visualization:\nImage generation was not successful.\n"
    
        # Creative Brief section
        creative_brief = "\nCreative Brief:\n"
        creative_points = [
            "Agency Goals and Objectives: Define our purpose and the narrative we'll present.",
            "Strategic Approach: Detail our thought process and market dynamics understanding.",
            "The Big Idea: Echo the central theme across all communication channels.",
            "Activation Mechanics System: Outline the campaign's rollout in line with strategic milestones.",
            "Production Plan: Map out logistical details for campaign content creation and dissemination.",
            "Media Plan and KPI Funnel: Set benchmarks for success and detail audience engagement tactics.",
            "Conclusion and Next Steps: Affirm our belief in the campaign and outline future developments."
        ]
        for point in creative_points:
            title, description = point.split(": ", 1)
            creative_brief += f"\n- **{title}:** {description}"
        final_presentation_content += creative_brief
    
        # Display the final presentation
        self.view.display_presentation(final_presentation_content)
    
        # Saving the session results with error checking
        save_result = self.save_session_results({
            'final_presentation': final_presentation_content,
            'product_image_url': product_image_url,
            'shopping_list': shopping_list_items
        })
        if not save_result:
            self.view.display_message("Failed to save session results.")
    
        return final_presentation_content
    
    
    
    
    
    
    def send_final_presentation(self, presentation_content, recipient_email):
        try:
            # Email sending logic here
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
    
            # Login Credentials for sending the email
            email_username = os.environ['EMAIL_USERNAME']
            email_password = os.environ['EMAIL_PASSWORD']
    
            if not email_username or not email_password:
                self.view.display_message("Email username or password not set.")
                return
    
            server.login(email_username, email_password)
    
            # Craft the email
            msg = MIMEMultipart()
            msg['From'] = email_username
            msg['To'] = recipient_email
            msg['Subject'] = "Final Presentation from AiMe"
    
            # Adding additional offer text to the email body
            additional_offer_text = (
                "\n\nIf you want to continue the advertisement for you — choose to do the next iteration focused on "
                "creating a Google Ads campaign. I can fine-tune Google Ads cabinet for your brand and start "
                "the campaign. Price 399€ + 10% from every conversion. You will see all processes in Google Analytics. "
                "Minimal time of the campaign is 3 months. And I work with an agreement only. Thank you for choosing AiMe."
            )
            presentation_content += additional_offer_text
    
            msg.attach(MIMEText(presentation_content, 'plain'))
    
            # Send the email
            server.send_message(msg)
            server.quit()
    
            self.view.display_message(f"Email sent successfully to {recipient_email}.")
        except smtplib.SMTPAuthenticationError as e:
            logging.error(f"SMTP Authentication failed: {e}")
            self.view.display_message("Failed to send email. Authentication error.")
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            self.view.display_message("An error occurred while sending the email.")
    
    
    def save_session_results(self, session_data, filename="session_results.json"):
        try:
            # Check and create the directory if it doesn't exist
            dir_name = os.path.dirname(filename)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
            # Initialize data
            data = []
    
            # If session_data contains a base64 image, save it to a file and replace the data with the file path
            if 'image_base64' in session_data and session_data['image_base64']:
                image_data = session_data['image_base64']
                image_path = os.path.join(dir_name, 'campaign_image.png')  # Define the path where the image will be saved
                with open(image_path, 'wb') as image_file:
                    image_file.write(base64.b64decode(image_data))
                session_data['image_base64'] = image_path  # Replace the base64 data with the file path
    
            # Check if file exists
            if os.path.isfile(filename):
                # Read existing data from the file
                with open(filename, 'r') as file:
                    data = json.load(file)
    
            # Append new session data to the existing data
            data.append(session_data)
    
            # Write the updated data back to the file
            with open(filename, 'w') as file:
                json.dump(data, file, indent=4)
    
            logging.info("Session results saved successfully.")
            return True
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}", exc_info=True)
            return False
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}", exc_info=True)
            return False
        except Exception as e:
            logging.error(f"An error occurred while saving session results: {e}", exc_info=True)
            return False
    
    
    
    
    
    
    
    
    
    
    
    
    # Other methods like handle_department_interaction, backtrack, etc., should be defined accordingly.
    