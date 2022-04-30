import difflib
import re


class DuplicateChecker:

    def __init__(self, fpath):
        self.duplicates = []
        self.fragment_separators = [".", "?", "!"]
        self.txt_files = {}
        self.read_txt(fpath)

    @staticmethod
    def clean_document(doc):
        l = doc.encode("ascii", "ignore").decode()
        # Remove all punctuation (i.e., quotes, commas, !, ?, etc.)
        l = re.sub(r"[^\w\s]", "", l)
        # Remove all Twitter handle mentions (i.e., “@UTK_EECS“ should be deleted.)
        l = re.sub(r"@[\w]+", " ", l)
        # Remove all instances of double-spaces.
        l = re.sub(r"\s+", " ", l)
        # Remove all 1-2 letter words
        l = re.sub(r"\W*\b\w{1,1}\b", "", l)
        # Convert all characters to lowercase.
        l = l.lower()
        # Remove all whitespace
        l = l.strip()

        return l

    def get_closest(self, prompt, cutoff):
        closest_matches = {}

        for k, v in self.txt_files.items():
            matches = difflib.get_close_matches(prompt, v, n=5, cutoff=cutoff)
            if len(matches) > 0:
                closest_matches[k] = matches

        if not closest_matches:
            return None
        return closest_matches

    def read_txt(self, file_path):
        print(f"# READING: {file_path}")
        self.txt_files[file_path] = set()
        with open(file_path, "r") as in_f:
            for doc in in_f.readlines():
                doc = self.clean_document(doc)
                self.txt_files[file_path].add(doc)

        print(f"=> {file_path}: {len(self.txt_files[file_path])} documents")

    def get_ratio(self, str_orig, str_match):
        return difflib.SequenceMatcher(None, str_match, str_orig).ratio()

    def check(self, text, threshold=0.75):
        for sep in self.fragment_separators:
            text = text.replace(sep, "\n")

        inp_frags = text.split("\n")
        for i1 in range(len(inp_frags)):
            inp_frags[i1] = self.clean_document(inp_frags[i1])
        inp_frags = list(filter(None, inp_frags))

        ratios = []
        results = {
            "Exact": 0,
            "Partial": 0,
            "Total": 0,
            "Fragments": {}
        }

        print("=" * 80)

        for i, frag in enumerate(inp_frags):
            close_matches = self.get_closest(frag, threshold)
            if close_matches is None:
                continue
            results["Fragments"][frag] = {}
            print(f"{i + 1}. {frag}")
            frag_results = {}
            for match, match_list in close_matches.items():
                mat_list = []
                for match_sentence in match_list:
                    ratio = self.get_ratio(match_sentence, frag)
                    if ratio == 1.0:
                        results["Exact"] += 1
                    else:
                        results["Partial"] += 1
                    ratios.append(ratio)
                    mat_list.append((match_sentence, ratio))
                    print(f"   {round(ratio * 100, 2)}% [{match}] ~ [{match_sentence}]")

                frag_results[match] = mat_list

            results["Fragments"][frag].update(frag_results)

        self.print_results(ratios, results)
        return results

    def print_results(self, ratios, results):
        results["Total"] = results["Exact"] + results["Partial"]
        results["Likelihood"] = sum(ratios) / len(ratios)
        print("=" * 80)
        print(f"# TOTAL FRAGMENTS: {results['Total']}")
        print(f"   => Exact: {results['Exact']} / {results['Total']} "
              f"({round(results['Exact'] / results['Total'] * 100, 2)}%)")
        print(f"   => Partial: {results['Partial']} / {results['Total']} "
              f"({round(results['Partial'] / results['Total'] * 100, 2)}%)")
        print(f"   => Total: {len(ratios)} / {results['Total']}"
              + f" ({round(len(ratios) / results['Total'] * 100, 2)}%)")
