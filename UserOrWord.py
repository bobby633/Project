from retrievetweets import Twitter, TwitterUser
#AIzaSyD054WimWA25uHzpbprc7RIIDj_6LlgcTU


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
