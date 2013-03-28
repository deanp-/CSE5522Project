# Your AI for CTF must inherit from the base Commander class.  See how this is
# implemented by looking at the commander.py in the ./api/ folder.
from api import Commander

# The commander can send 'Commands' to individual bots.  These are listed and
# documented in commands.py from the ./api/ folder also.
from api import commands

# The maps for CTF are layed out along the X and Z axis in space, but can be
# effectively be considered 2D.
from api import Vector2
#from array import array
#from numpy import array

from scipy import *
from pybrain.rl.environments.mazes import Maze, MDPMazeTask
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import Experiment
from pybrain.rl.environments import Task

class TestCommander(Commander):
	"""Initial test commander for basic reinforcement learning"""

	teamName = ""
	goalLoc = Vector2(0,0)
	respawnTime = 0
	maxRespawnTime = 45

	def initialize(self):
		#grabs initial game state for the game
		gameState = self.game
		#grabs initial level state for the level
		lvlState = self.level

		#creates the binary representation of the map
		structure = array(self.level.blockHeights)
		for line in structure:
			for element in range(len(line)):
				if line[element] > 0:
					line[element] = 1

		nm = self.name

		"""information from the game state"""
		matchInfo = gameState.match
		teamInfo = gameState.teams
		myInfo = gameState.team
		enemyInfo = gameState.enemyTeam
		botInfo = gameState.bots
		flagInfo = gameState.flags

		aliveBots = gameState.bots_alive
		availableBots = gameState.bots_available
		holdingBots = gameState.bots_holding
		enemyFlagInfo = gameState.enemyFlags
		teamName = myInfo.name
                self.respawnTime = matchInfo.timeToNextRespawn
    
		"""Initialize reinforcement learning"""
		goalPoint = teamInfo[teamName].flagSpawnLocation
		environment = Maze(structure, (goalPoint.x,goalPoint.y))
		totalArea = lvlState.height * lvlState.width
		controller = ActionValueTable(totalArea, 4)
		controller.initialize(1.)
		learner = Q()
		agent = LearningAgent(controller, learner)
		task = MDPMazeTask(environment)
		experiment = Experiment(task, agent)
		#experiment.doInteractions(1)
		#agent.learn()
		#agent.reset()

	def tick(self):

                #global feature vector
		fv = self.getFeatureVector()
		#feature vectors for each bot
		bfvs = self.getBotFeatureVectors()
		
		#print bfvs
		#print fv
		
                for bot in self.game.bots_available:

                        #output from neural net determines bot action
                        botActionChoice = 0

                        if botActionChoice == 0:
                                if bot.flag:
                                # go to score location
                                        flagScoreLocation = self.game.team.flagScoreLocation
                                        self.issue(commands.Charge, bot, flagScoreLocation, description = 'Run to my flag')
                                else:
                                # go to enemy flag location
                                        enemyFlag = self.game.enemyTeam.flag.position
                                        self.issue(commands.Charge, bot, enemyFlag, description = 'Run to enemy flag')
                                        
                        elif botActionChoice == 1:
                                # attack nearest enemy
                                nearestEnemyBot = None
                                enBotPos = None
                                nearestBotDist = 100000.0
                                for enBot in bot.visibleEnemies:
                                        if bot.position.distance(enBot.position) < nearestBotDist:
                                                nearestEnemyBot = enBot
                                                enBotPos = enBot.position
                                                nearestBotDist = bot.position.distance(enBot.position)
                                if nearestEnemyBot != None:
                                        self.issue(commands.Attack, bot, enBotPos, enBotPos, description = 'Attack nearest enemy')
                                        
                        elif botActionChoice == 2:
                                #charge nearest enemy
                                nearestEnemyBot = None
                                enBotPos = None
                                nearestBotDist = 100000.0
                                for enBot in bot.visibleEnemies:
                                        if bot.position.distance(enBot.position) < nearestBotDist:
                                                nearestEnemyBot = enBot
                                                enBotPos = enBot.position
                                                nearestBotDist = bot.position.distance(enBot.position)
                                if nearestEnemyBot != None:
                                        self.issue(commands.Charge, bot, nearestEnemyBot, enBotPos, description = 'Charge nearest enemy')

                        elif botActionChoice == 3:       
                        #defend current position
                               self.issue(commands.Defend, None, description = 'Defend position')

                        elif botActionChoice == 4:
                        #defend flag
                                flagPosition = myInfo.flag.position
                                if bot.position.distance(flagPosition) > 2:
                                        self.issue(commands.Attack, bot, flagPosition, flagPosition, description = 'Move to flag location')
                                else:
                                        self.issue(commands.Defend, None, description = 'Defend the flag')
                                        

                        elif botActionChoice == 5:
                        #escort flag carrier
                                pass
                        elif botActionChoice == 6:
                        #attack enemy flag carrier
                                pass

                                
	def shutdown(self):
		pass
	
	def getFeatureVector(self):
                """
                        Gets the global feature vector for the game
                        Returns a list of numeric values
                """
                
		#grabs initial game state for the game
		gameState = self.game
		#grabs initial level state for the level
		lvlState = self.level

		"""information from the game state"""
		matchInfo = gameState.match
		teamInfo = gameState.teams
		myInfo = gameState.team
		enemyInfo = gameState.enemyTeam
		botInfo = gameState.bots
		flagInfo = gameState.flags	

		aliveBots = gameState.bots_alive
		availableBots = gameState.bots_available
		holdingBots = gameState.bots_holding
		enemyFlagInfo = gameState.enemyFlags

		#team names
		teamName = myInfo.name
		enemyName = enemyInfo.name

		#is a bot carrying the flag
		flagCarried = 0
		enFlag = enemyInfo.flag.name
		if(enemyInfo.flag.carrier != None):
			flagCarried = 1
		#is our flag taken
		ourFlagCarried = 0
		ourFlag = myInfo.flag.name
		if(myInfo.flag.carrier != None):
			ourFlagCarried = 1
		#distance from their flag carrier from our score point
		myFlagTotalDist = enemyInfo.flagSpawnLocation.distance(myInfo.flagScoreLocation)
		myFlagDist = enemyInfo.flag.position.distance(myInfo.flagScoreLocation)
		myFlagDist = myFlagDist/myFlagTotalDist
		#distance from our flag carrier to their score point
		theirFlagTotalDist = myInfo.flagSpawnLocation.distance(enemyInfo.flagScoreLocation)
		theirFlagDist = myInfo.flag.position.distance(enemyInfo.flagScoreLocation)
		theirFlagDist = theirFlagDist/theirFlagTotalDist
		#total respawn time (max respawn time is 45 seconds from observation)
		respawnTime = self.respawnTime/self.maxRespawnTime
		#time until respawn
		timeToRespawn = matchInfo.timeToNextRespawn/lvlState.respawnTime
		#percentage of time remaining in the game
		timeR = matchInfo.timeRemaining
		timeP = matchInfo.timePassed
		timeLeft = timeR/(timeR+timeP)
		#number of bots on the team
		totalBotNum = (.5*len(gameState.bots))/15.0
		#percentage of friendly bots alive
		totalBots = len(gameState.bots_alive)/(.5*len(gameState.bots))
		#percentage of enemy bots seen
		totalEnemy = set([])
		for bot in gameState.bots_alive:
			totalEnemy = totalEnemy.union(bot.visibleEnemies)
		enemyNumber = len(totalEnemy)/(.5*len(gameState.bots))
		#score differential(0 losing, .5 tied, 1 winning)
		myScore = matchInfo.scores[teamName]
		enemyScore = matchInfo.scores[enemyName]
		scoreDiff = .5
		if(myScore > enemyScore):
			scoreDiff = 1
		elif(enemyScore > myScore):
			scoreDiff = 0
		elif(enemyScore == myScore):
			scoreDiff = .5

                #density of moveable and visible area
		mapStruct = lvlState.blockHeights
		totalArea = lvlState.width * lvlState.height
                moveDensity = 0.0
                visibleDensity = 0.0
		for row in mapStruct:
                        for block in row:
                                if block == 0:
                                        moveDensity = moveDensity + 1
                                        visibleDensity = visibleDensity + 1
                                elif block < 3:
                                        visibleDensity = visibleDensity + 1

		moveDensity = moveDensity/totalArea
		visibleDensity = visibleDensity/totalArea

                #time until friendly flag despawns
		if myInfo.flag.respawnTimer < 0:
                        ourFlagDespawn = 0
                else:
                        ourFlagDespawn = myInfo.flag.respawnTimer/30.0
		#time until enemy flag despawns
		if enemyInfo.flag.respawnTimer < 0:
                        enemyFlagDespawn = 0
                else:
                        enemyFlagDespawn = enemyInfo.flag.respawnTimer/30.0
		
		#Global feature vector for the game
		featureVector = [
				flagCarried, 
				ourFlagCarried,
				myFlagDist, 
				theirFlagDist,
                                respawnTime,
				timeToRespawn, 
				timeLeft,
                                totalBotNum,
				totalBots, 
				enemyNumber, 
				scoreDiff,
                                moveDensity,
                                visibleDensity,
                                ourFlagDespawn,
                                enemyFlagDespawn]
		return featureVector
	
	def getBotFeatureVectors(self):
                """
                        Gets the feature vector for individual bots on the team
                        Returns a dictionary containing the bot names and their
                        corresponding feature vectors
                        i.e. {"red1": [1,0,1,....], "red2": [0,1,1,....], ......}
                """
                
                fvDict = {}

		#grabs initial game state for the game
		gameState = self.game
		#grabs initial level state for the level
		lvlState = self.level

		#normalizing distance (farthest distance possible, 1 corner to the other)
		#refactor to more global scope later (value does not change)
		d1 = lvlState.area[0]
                d2 = lvlState.area[1]
                normDist = d1.distance(d2)
                
		"""information from the game state"""
		matchInfo = gameState.match
		teamInfo = gameState.teams
		myInfo = gameState.team
		enemyInfo = gameState.enemyTeam
		botInfo = gameState.bots
		flagInfo = gameState.flags
		
                #generate feature vector for each bot
                for bot in myInfo.members:

                        #current position of the bot
                        botPos = bot.position
                        
                        #HasFlag
                        hasFlag = 0
                        if bot.flag != None:
                                hasFlag = 1
                        #Health
                        isAlive = 0
                        if bot.health > 0:
                                isAlive = 1
                        #See enemy
                        seesEnemy = 0
                        if len(bot.visibleEnemies) > 0:
                                seesEnemy = 1
                        #Seen by enemy
                        seenByEnemy = 0
                        if len(bot.seenBy) > 0:
                                seenByEnemy = 1
                        #distance to enemy FC
                        distToEnemyFC = 0
                        if myInfo.flag.carrier != None:
                                distToEnemyFC = botPos.distance(myInfo.flag.carrier.position)/normDist
                        #distance to our flag/FC
                        distToFriendlyFC = 0
                        if enemyInfo.flag.carrier != None:
                                distToFriendlyFC = botPos.distance(enemyInfo.flag.carrier.position)/normDist
                        #distance to nearest seen enemy
                        distToNearestEnemy = 1
                        if len(bot.visibleEnemies) > 0:
                                for enBot in bot.visibleEnemies:
                                        tDist = botPos.distance(enBot.position)/normDist
                                        if tDist < distToNearestEnemy:
                                                distToNearestEnemy = tDist
                        #distance to nearest ally
                        distToNearestAlly = 1
                        for frBot in myInfo.members:
                               if frBot != bot:
                                       tDist = botPos.distance(frBot.position)/normDist
                                       if tDist < distToNearestAlly:
                                               distToNearestAlly = tDist

                        """NOT WORKING, NOT INCLUDED IN VECTOR"""
                        #distance to combat/ally support
                        distToNearestCombat = 1
                        for btBot in gameState.bots_holding:
                                tDist = botPos.distance(btBot.position)/normDist
                                if tDist < distToNearestCombat:
                                        distanceToNearestCombat = tDist

                        #distance to friendly spawn zone
                        mySpawn = myInfo.botSpawnArea
                        distToFriendlySpawn = botPos.distance(mySpawn[0].midPoint(mySpawn[1]))/normDist
                        #distance to enemy spawn zone
                        enemySpawn = enemyInfo.botSpawnArea
                        distToEnemySpawn = botPos.distance(enemySpawn[0].midPoint(enemySpawn[1]))/normDist

                       
                        #feature vector for individual bot
                        botFV = [
                                hasFlag,
                                isAlive,
                                seesEnemy,
                                seenByEnemy,
                                distToEnemyFC,
                                distToFriendlyFC,
                                distToNearestEnemy,
                                distToNearestAlly,
                                distToFriendlySpawn,
                                distToEnemySpawn]

                        fvDict[bot.name] = botFV
                        
                return fvDict
