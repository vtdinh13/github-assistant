import ingest
import search_agent 
import logs
import argparse

import asyncio



def initialize_index(repo_owner:str, repo_name:str):
    print(f"Starting AI Assistant for {repo_owner}/{repo_name}")
    print("Initializing data ingestion...")

    def filter(doc):
        return 'data-engineering' in doc['filename']

    index = ingest.index_data(repo_owner, repo_name, filter=filter)
    print("Data indexing completed successfully!")
    return index


def initialize_agent(index, repo_owner:str, repo_name:str):
    print("Initializing search agent...")
    agent = search_agent.init_agent(index, repo_owner, repo_name)
    print("Agent initialized successfully!")
    return agent


def main(params):
    repo_owner = params.repo_owner
    repo_name = params.repo_name
    index = initialize_index(repo_owner=repo_owner, repo_name=repo_name)
    agent = initialize_agent(index, repo_owner=repo_owner, repo_name=repo_name)
    print("\nReady to answer your questions!")
    print("Type 'stop' to exit the program.\n")

    while True:
        question = input("Your question: ")
        if question.strip().lower() == 'stop':
            print("Goodbye!")
            break

        print("Processing your question...")
        response = asyncio.run(agent.run(user_prompt=question))
        logs.log_interaction_to_file(agent, response.new_messages())

        print("\nResponse:\n", response.output)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an agent that is grounded with data from a given repository. Embeddings, created from texts, are stored in vectors and later retrieved when a question is posed to the agent.')
    parser.add_argument('--repo_owner', help='user id of repository owner')
    parser.add_argument('--repo_name', help='name of repository')

    args = parser.parse_args()
    main(args)