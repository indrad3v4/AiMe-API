import spacy
from spacy import displacy
from spacy.matcher import Matcher
import json
import base64
import re
import logging
from roles import *

# Before the AdAgency class, load the spaCy model
nlp = spacy.load("en_core_web_md")


class AdAgency:  
    def __init__(self, name):
        self.name = name
        self.departments = {}

    def add_department(self, department):
        self.departments[department.name] = department

    def get_department(self, name):
        return self.departments.get(name)


class Department:
    def __init__(self, name):
        self.name = name
        self.roles = []
  
    def add_role(self, role):
        self.roles.append(role)
  
    def get_role(self, role_name):
        for role in self.roles:
            if role.name == role_name:
                return role
        return None

class Role:
    def __init__(self, name, description, responsibilities, controller):
        self.name = name
        self.description = description
        self.responsibilities = responsibilities
        self.feedback_history = []
        self.previous_responses = [] 
        self.matcher = Matcher(nlp.vocab)
        self.controller = controller
  
    def analyze_client_brief(self, client_brief):
        doc = nlp(client_brief)
        entities = []
  
        for ent in doc.ents:
            entity_info = {
              "text": ent.text,
              "label": ent.label_,
              "adjectives": [],
              "verbs": [],
              "sentiment": None,
              "related_words": []
            }
  
            # Analyze sentiment of the sentence where the entity is mentioned
            sentence = ent.sent.root.sent
            entity_info["sentiment"] = TextBlob(sentence.text).sentiment.polarity
  
            # Find modifiers (adjectives) and actions (verbs) related to the entity
            if token.head == ent.root:
                if token.pos_ == "ADJ":
                    entity_info["adjectives"].append(token.text)
                elif token.pos_ == "VERB":
                    entity_info["verbs"].append(token.lemma_)
  
              # Add context by considering nearby noun phrases and chunks
            entity_info["context"] = [chunk.text for chunk in sentence.noun_chunks if ent.root in chunk]
  
            # Dependency parsing to understand the context around entities
            for token in ent.root.subtree:
                if token.dep_ in ("amod", "prep", "conj", "nmod"):
                    entity_info["related_words"].append(token.text)
                if token.dep_ == "acomp":
                    entity_info["adjectives"].append(token.text)
  
            entities.append(entity_info)
  
        return entities
  
    def generate_shopping_list(self, ai_response, client_data):
        # Use the existing AI response to get context for the shopping list
        context = f"Based on the following details, {ai_response}, create a detailed shopping list for the campaign."
  
        # Now call the OpenAI API to get a shopping list based on the context
        shopping_list_prompt = self.create_shopping_list_prompt(client_data)
        shopping_list_response = self.controller.call_openai_api(shopping_list_prompt)
  
        # Parse the returned shopping list into a structured format
        shopping_list_items = self.parse_shopping_list(shopping_list_response)
  
        return shopping_list_items
  
    def create_shopping_list_prompt(self, client_data):
        # Create a prompt that describes the client's campaign and asks for a shopping list
        prompt = f"Generate a shopping list for an advertising campaign targeting {client_data['target_audience']} with the goal of {client_data['goals']}. The campaign is for the product {client_data['product']} and should reflect the message '{client_data['message']}' and the big idea '{client_data['big_idea']}'."
        return prompt
  
    def parse_shopping_list(self, response):
        # Parse the response into a list of items
        shopping_list = OrderedDict()
        doc = nlp(response)
        for sentence in doc.sents:
            for token in sentence:
                # Look for verbs that suggest an action item or goal
                if token.pos_ == "VERB" and token.lemma_ in ["buy", "include", "need", "require"]:
                    # Look for direct objects of the verb (things to buy/include/need/require)
                    for child in token.children:
                        if child.dep_ == "dobj":
                            # Get compound nouns or adjectives connected to the direct object
                            item_components = [child.text] + [w.text for w in child.children if w.dep_ in ["compound", "amod"]]
                            item = " ".join(item_components)
                            shopping_list[item] = None  # Use None as a placeholder
        return list(shopping_list.keys())
  
  
    def generate_response(self, input_data, client_brief, last_feedback, client_data):
        self.previous_responses = []
        # Analyze the client brief to get entities and their related context
        entities_info = self.analyze_client_brief(client_brief)
  
        # Construct insights string from entities and their related words for the creative prompt
        insights = "; ".join([f"{entity['text']} ({', '.join(entity['related_words'])})"
                              for entity in entities_info if entity['related_words']])
  
        # Determine the overall sentiment from the entities' sentiments
        sentiment_summary = 'positive' if all(entity['sentiment'] > 0 for entity in entities_info if entity['sentiment'] is not None) else 'mixed'
  
        # Include previous responses for context continuity
        previous_responses_str = "\n\nPrevious responses:\n" + "\n".join(self.previous_responses) if self.previous_responses else ""
  
        # If an image is provided, add a reference to it in the response
        image_context = ""
        if client_data.get('image_base64'):
            image_context = "An image has been provided to enrich the campaign's visual context and will be integrated into our creative strategies."
  
        # Construct role-specific questions based on the role's responsibilities
        role_specific_questions = f"Considering your role's focus on {self.responsibilities}, how would you leverage these insights and the provided image? What innovative strategies would you propose to achieve {client_data['goals']}?"
  
        # Enhanced prompt with entities, their context, sentiments, previous responses, and image context
        creative_prompt = f"As a {self.name}, inspired by {self.description}, your task is to develop a part of the campaign that aligns with the client's vision. Insights from the client brief include: {insights}. The overall sentiment is {sentiment_summary}. {image_context} Considering the last feedback: {last_feedback}, and previous discussions: {previous_responses_str}, {role_specific_questions}"
  
        # Call the general-purpose API interaction method
        response_content = self.controller.call_openai_api(
            creative_prompt,
            client_data=client_data
        )
  
        # Format the response as Markdown
        response_content_md = f"## Response from {self.name}\n\n" \
                              f"### Insights\n" \
                              f"{insights}\n\n" \
                              f"### Sentiment Analysis\n" \
                              f"Sentiment is {sentiment_summary}.\n\n" \
                              f"### Response\n" \
                              f"{response_content}\n\n" \
                              f"{image_context}"
  
        # Append the Markdown-formatted response to previous responses for future context
        if "Unable to generate response" not in response_content:
            self.previous_responses.append(response_content_md)
  
            # Optionally, generate a shopping list or other actionable items from the response
            shopping_list = self.generate_shopping_list(response_content, client_data) if hasattr(self, 'generate_shopping_list') else []
        else:
            # Handle error by logging and setting shopping_list to empty
            logging.error("Failed to generate response from OpenAI API.")
            response_content_md = "An error occurred while generating the response."
            shopping_list = []
  
        return response_content_md, shopping_list
  
  
  
  
    def store_feedback(self, feedback):
        # Store feedback in the feedback history
        self.feedback_history.append(feedback)
  