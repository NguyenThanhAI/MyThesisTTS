from text import text_to_sequence, cmudict

if __name__ == "__main__":
    cmu_dict = cmudict.CMUDict(file_or_path="resources/cmu_dictionary")
    txt = "Hello. My name is Thanh"
    phonemes = text_to_sequence(text=txt, dictionary=cmu_dict)
    print("Phonemes: {}".format(phonemes))