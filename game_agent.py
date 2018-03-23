import random

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2*opp_moves)


def custom_score_2(game, player):   # similar to custom_score but favors blocking opponents move after the board is half full
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    booster=0 #
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    if game.move_count>(game.width)*(game.height)/2:
        pref_moves=list(set(own_moves) & set(opp_moves))
        if len(pref_moves):
            booster=1
    return float(len(own_moves) - 2*len(opp_moves)+8*booster)    # favors moves that limit opponent's choice


def custom_score_3(game, player): # in the begining of the game (for first 18 moves on 7*7 sized board) favors cells that are surrounded by blocked cells. 
    if game.is_loser(player):    # after first 18 moves seeks for blocking opponent move. together with improved_score.
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    occupied_neighbors_amount=0
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    if game.move_count<(game.width-4)*(game.height-4)*2:  
        x,y=game.get_player_location(player)
        if x==0: xx=[0,1]
        elif x==game.width-1: xx=[game.width-2, game.width-1]
        else: xx=[x-1,x,x+1]
        if y==0: yy=[0,1]
        elif y==game.height-1: yy=[game.height-2, game.height-1]
        else: yy=[y-1,y,y+1]
        non_empty=[(i, j) for i in xx for j in yy if game._board_state[i + j * game.height] != 0]
        non_empty.remove((x,y))
        occupied_neighbors_amount=len(non_empty) # non-empty spaces
    else:
        if len(opp_moves)==1 and opp_moves[0] in own_moves: occupied_neighbors_amount=float('inf')
    return float(len(own_moves) - len(opp_moves)+6*occupied_neighbors_amount)


class IsolationPlayer:

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):


    def get_move(self, game, time_left):
        self.time_left = time_left
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves=game.get_legal_moves(self)
        if len(legal_moves):
            _, move=max([(self.score(game.forecast_move(m), self),m) for m in legal_moves])#argmax for scores at last depth
            best_move= move
        else: best_move=(-1,-1)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move=self.minimax(game, self.search_depth)
            return best_move
        except SearchTimeout:
            pass 

    def minimax(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        legal_moves=game.get_legal_moves(self) 
        if len(legal_moves)==0:
            return (-1,-1)
        _, move=max([(self.min_value(game.forecast_move(m), depth-1),m) for m in legal_moves]) # argmax for recursive calls 
        return move

    def max_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()	
        if len(game.get_legal_moves(self))==0: #it appears as redundant check, but it does expedite solutions in some cases and improves tournament score
            return self.score(game, self)
        if depth:
            v=float('-inf')
            for a in game.get_legal_moves(self):  # maximizing player
                v=max(v, self.min_value(game.forecast_move(a), depth-1))
            return v
        else: return self.score(game, self)

    def min_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if len(game.get_legal_moves(game.get_opponent(self)))==0:
            return self.score(game, game.get_opponent(self))
        if depth:
            v=float('inf')
            for a in game.get_legal_moves(game.get_opponent(self)):   #minimizing player
                v=min(v, self.max_value(game.forecast_move(a), depth-1))
            return v
        else: return self.score(game, self)

class AlphaBetaPlayer(IsolationPlayer):

    def get_move(self, game, time_left):
        self.time_left = time_left
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves=game.get_legal_moves(self)
        if len(legal_moves):
            _, move=max([(self.score(game.forecast_move(m), self),m) for m in legal_moves])#argmax for scores at last depth
            best_move= move
        else: 
            return (-1,-1)
        try:
            for self.search_depth in range(1, game.height*game.width+1):
                best_move=self.alphabeta(game, self.search_depth)
        except SearchTimeout:
            pass 
        return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        max_v=float('-inf')
        for m in game.get_legal_moves(self):
            v=self.min_value(game.forecast_move(m), depth-1, alpha, beta)
            if v>=max_v:
                max_v=v
                best_move=m
            alpha=max(alpha, v)
        return best_move


    def max_value(self, game, depth, alpha, beta):
        v=float('-inf')
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()	
        if len(game.get_legal_moves(self))==0:
            return self.score(game, self)
        if depth:
            for a in game.get_legal_moves(self):  #maximizing player
                v=max(v, self.min_value(game.forecast_move(a), depth-1, alpha, beta))
                if v>= beta: 
                    return v
                alpha=max(alpha, v)
            return v
        else: return self.score(game, self)      

    def min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        v=float('inf')
        if len(game.get_legal_moves(game.get_opponent(self)))==0:
            return self.score(game, self)
        if depth:
            for a in game.get_legal_moves(game.get_opponent(self)):   #minimizing player
                v=min(v, self.max_value(game.forecast_move(a), depth-1, alpha, beta))
                if v<=alpha: return v
                beta=min(beta, v)
            return v
        else: return self.score(game, self)
