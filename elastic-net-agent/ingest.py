import io
import zipfile
import requests
import frontmatter
import argparse
import pickle
from tqdm.auto import tqdm

def read_repo_data(repo_owner:str, repo_name:str) -> list:
    """
    Download and parse all markdown files from a GitHub repository.
    
    Args:
        repo_owner: GitHub username or organization
        repo_name: Repository name
    
    Returns:
        List of dictionaries containing file content and metadata
    """
    prefix = 'https://codeload.github.com' 
    url = f'{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)
    
    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    
    for file_info in zf.infolist():
        filename = file_info.filename.lower()

        if not filename.endswith(('.md', '.mdx')):
            continue
    
        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode('utf-8', errors='ignore')
                post = frontmatter.loads(content)
                data = post.to_dict()
                data['filename'] = filename
                repository_data.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    zf.close()

    
    with open('git_repo_data.pkl', 'wb') as f:
        pickle.dump(repository_data, f)
    return repository_data   

def main(params):
    repo_owner = params.repo_owner
    repo_name = params.repo_name
    read_repo_data(repo_owner, repo_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest git repo and save data as a pickle file in current working directory')
    parser.add_argument('--repo_owner', help='user id of repository owner')
    parser.add_argument('--repo_name', help='name of repository')

    args = parser.parse_args()
    main(args)