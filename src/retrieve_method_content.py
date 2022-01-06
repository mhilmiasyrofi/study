import os, glob, requests
import pandas as pd
from tree_sitter import Language, Parser

Language.build_library(
    # Store the library in the `build` directory
    'build/my-languages.so',

    # Include one or more languages
    [
        '/Users/mhilmiasyrofi/Documents/AI4SAWI/tree-sitter-java'
    ]
)

JAVA_LANGUAGE = Language('build/my-languages.so', 'java')

parser = Parser()
parser.set_language(JAVA_LANGUAGE)

def convert_github_url_into_raw_url(github_url:str) -> str:
    """Convert github url into raw github url.
    Args
        github_url: github url
    Returns
        raw github url
    """
    github_url = github_url.split("#L")[0]
    raw_github_url = github_url.replace("github.com", "raw.githubusercontent.com"). \
        replace("/tree/", "/")
    return raw_github_url


def retrieve_method_content(github_url:str, target_method_name:str) -> str:
    """Retrieve method content given github url and target method name
    Args:
        github_url:
        target_method_name:
    Returns:
        method content from the source code

    Details:
        - convert github url into raw github url
        - create request to the github url, 
        - if the response is valid, parse the code and find the snippet code containg the target_method_name
        - otherwise return an empty string
    """

    print(github_url)
    print(target_method_name)
    raw_github_url = convert_github_url_into_raw_url(github_url)

    response = requests.get(raw_github_url)

    if response.status_code == 200:
        code_bytes = response.content
    else:
        print("Github URL: ", github_url)
        print("Raw Github URL: ", raw_github_url)
        # raise ValueError("No response from Github")
        return ""

        ## found example code that return 404 not found
        ## https://github.com/apache/cassandra/tree/4ed2234078c4d302c256332252a8ddd6ae345484//src/gen-java/org/apache/cassandra/cql/CqlParser.java#L3431

    tree = parser.parse(code_bytes)

    query = JAVA_LANGUAGE.query("""
    (method_declaration 
      name: (identifier) 
      body: (block) 
    ) @method_decl
    """)

    captures = query.captures(tree.root_node)

    res = ""

    for capture in captures:
        node, node_type = capture
        name_node = node.child_by_field_name('name')

        method_name = code_bytes[name_node.start_byte:name_node.end_byte].decode(
            'utf8')
        if method_name == target_method_name:
            node_text = code_bytes[node.start_byte:node.end_byte].decode(
                'utf8')
            # print(method_name, node_text)
            res = node_text
            break

    return res

# def test_retrieve_method_content():



if __name__ == "__main__" :

    dir_to_csv_files = "/Users/mhilmiasyrofi/Documents/AI4SAWI/study/data/test"

    output_dir = dir_to_csv_files + "-derived"

    os.makedirs(output_dir,exist_ok=True)
    

    for file_path in glob.iglob(dir_to_csv_files + '/**/*.csv', recursive=True):
        print(f"Processing {file_path}")
        df = pd.read_csv(file_path)

        df["method_content"] = df.apply(
            lambda x:  retrieve_method_content(x["url"], x["method_name"]) if (x["url"] != "url" and x["method_name"] != None) else "", axis=1)

        df = df[df["method_content"] != ""]
        
        output_file = file_path.replace(dir_to_csv_files, output_dir)

        df.to_csv(output_file, index=False)



