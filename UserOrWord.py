from retrievetweets import Twitter, TwitterUser



class SearchUserorWord:
    def pickUserorWord():
        print("User or keyword ?")
        choice = input()
        if choice == "u" :
            TwitterUser.twitter_user()
        elif choice == "k" :
            Twitter.twitter()
        else:
            print("wrong")