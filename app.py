from UserOrWord import SearchUserorWord
from getreddit import Reddit


class Main:
    def execute(choice):
        if choice == "t":
            SearchUserorWord.pickUserorWord()
            print ("error")
        elif choice == "r":
            Reddit.get_reddit()       
        else:
            print ("error")

if __name__== "__main__":  
    print("twitter or reddit?")
    choice = input()
    Main.execute(choice)