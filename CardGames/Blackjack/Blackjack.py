import random
suits = ['Spades', 'Clubs', 'Hearts', 'Diamonds']
ranks = ['Ace', 'Two', 'Three', 'Four', 'Five' ,'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']
values = {'Ace': 11, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5 ,'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9, 
          'Ten': 10, 'Jack': 10, 'Queen': 10, 'King': 10}

class Card():
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
    
    def __str__(self):
        return self.rank + ' of ' + self.suit
    
class Deck():
    def __init__(self):
        self.deck = []
        for suit in suits:
            for rank in ranks:
                self.deck.append(Card(suit, rank))
                
    def __str__(self):
        deck_comp = ''
        for card in self.deck:
            deck_comp += '\n ' + card.__str__()
        return 'The deck has:' + deck_comp 
    
    def shuffle(self):
        random.shuffle(self.deck)
        
    def deal(self):
         return self.deck.pop(0)
        
class Hand():
    def __init__(self):
        self.cards = []
        self.value = 0
        self.aces = 0
    
    def add_card(self, card):
        self.cards.append(card)
        self.value += values[card.rank]
        if card.rank == 'Ace':
            self.aces += 1
    
    def adjust_for_ace(self):
        while self.value > 21 and self.aces > 0:
            self.value -= 10
            self.aces -= 1
            
class Chips():
    def __init__(self, total = 100):
        self.total = total
        self.bet = 0
    
    def win_bet(self):
        self.total += self.bet
        
    def lose_bet(self):
        self.total -= self.bet
        
#How much to bet
def take_bet(chips):
    while True:
        try:
            chips.bet = int(input('How many chips would you like to bet? '))
        except:
            print('Your bet must be a positive integer')
        else:
            if chips.bet > chips.total:
                print("Your bet can't exceed your total")
            else:
                break
                

#Actions to take
def hit(deck, hand):
    hand.add_card(deck.deal())
    hand.adjust_for_ace()
    
    
def hit_or_stand(deck, hand):
    global playing
    
    while True:
        x = input("\nWould you like to hit or stand? Enter 'h', or 's'")
        
        if x[0].lower() == 'h':
            hit(deck, hand)
                
        elif x[0].lower() == 's':
            print('Player stands. Dealer is playing')
            playing = False
            break
        
        else:
            print("Try again")
            continue
            
        break
        
#Player and dealer are hand classes. 
#Showing cards in hand

def show_some(player, dealer):
    print("\nDealer's Hand:")
    print(' <Card Hidden>')
    print('', dealer.cards[1])
    print("\nPlayer's Hand:", *player.cards, sep = '\n ')
    
def show_all(player, dealer):
    print("\nDealer's Hand:", *dealer.cards, sep = '\n ')
    print("Dealer's Points:", dealer.value)
    print("\nPlayer's Hand:", *player.cards, sep = '\n ')
    print("Player's Points:", player.value)
    
#Outcome    
def player_bust(player, dealer, chips):
    print("Player busts!")
    chips.lose_bet()
    
def player_wins(player, dealer, chips):
    print("Player wins!")
    chips.win_bet()
    
def dealer_bust(player, dealer, chips):
    print("Dealer busts!")
    chips.win_bet()
    
def dealer_wins(player, dealer, chips):
    print("Dealer wins!")
    chips.lose_bet()
    
def push(player, dealer):
    print("Player and Dealer tie. It's a push!")
    

