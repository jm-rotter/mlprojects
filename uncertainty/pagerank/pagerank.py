import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    if len(corpus[page])  == 0 and damping_factor != 0:
        return transition_model(corpus, page, 0)


    
    length = len(corpus.keys())

    result = {}
     
    for key in corpus.keys():
        result.update({key: (1-damping_factor) * 1/length})

    length = len(corpus[page])

    for value in corpus[page]:
        result[value] += damping_factor * 1/length 
   
    return result
    


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    res_dict = {key:0 for key in corpus.keys()} 
    curr_page = random.choice(list(corpus.keys())) 
    res_dict[curr_page] += 1


    for i in range(n):
        new_dict = transition_model(corpus, curr_page, damping_factor)
        curr_page = random.choices(list(new_dict.keys()), list(new_dict.values()), k =1)[0]
        res_dict[curr_page] += 1
    
    return {key:res_dict[key]/(n+1) for key in corpus.keys()}
        

def convergence(conv,npage, oldpage):
    for key in npage:
        if conv < abs(npage[key] - oldpage[key]):
            return False
    
    return True

def linking(page, corpus):
    return {match for match in corpus if page in corpus[match] or len(corpus[match]) == 0}

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    res_dict = {key:1/N for key in corpus} 

    while(True):
        npage = {}
        for key in corpus.keys():
            kr = (1-damping_factor)/N 
            sum_pr = 0

            for page in linking(key, corpus):
                if len(corpus[page]) == 0:
                    sum_pr += res_dict[page] / N 
                else:
                    sum_pr += res_dict[page] / len(corpus[page])


            npage[key] = kr + damping_factor*sum_pr

        if convergence(0.001,npage, res_dict):
            return npage
        res_dict = npage
                

if __name__ == "__main__":
    main()
