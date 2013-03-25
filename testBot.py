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
		fv = self.getFeatureVector()
		print fv
		
        	for bot in self.game.bots_available:
            		if bot.flag:
                	# if a bot has the flag run to the scoring location
                		flagScoreLocation = self.game.team.flagScoreLocation
                		self.issue(commands.Charge, bot, flagScoreLocation, description = 'Run to my flag')
            		else:
                	# otherwise run to where the flag is
                		enemyFlag = self.game.enemyTeam.flag.position
                		self.issue(commands.Charge, bot, enemyFlag, description = 'Run to enemy flag')
			
	def shutdown(self):
		pass
	
	def getFeatureVector(self):
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
		#distance between the bots and enemy flag
		#distance between the bots and the team's flag
		#distance from their flag carrier from our score point
		myFlagTotalDist = enemyInfo.flagSpawnLocation.distance(myInfo.flagScoreLocation)
		myFlagDist = enemyInfo.flag.position.distance(myInfo.flagScoreLocation)
		myFlagDist = myFlagDist/myFlagTotalDist
		#distance from our flag carrier to their score point
		theirFlagTotalDist = myInfo.flagSpawnLocation.distance(enemyInfo.flagScoreLocation)
		theirFlagDist = myInfo.flag.position.distance(enemyInfo.flagScoreLocation)
		theirFlagDist = theirFlagDist/theirFlagTotalDist
		#time until respawn
		respawnTime = matchInfo.timeToNextRespawn/lvlState.respawnTime
		#percentage of time remaining in the game
		timeR = matchInfo.timeRemaining
		timeP = matchInfo.timePassed
		timeLeft = timeR/(timeR+timeP)
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

		
		featureVector = [
				flagCarried, 
				ourFlagCarried,
				myFlagDist, 
				theirFlagDist,
				respawnTime, 
				timeLeft, 
				totalBots, 
				enemyNumber, 
				scoreDiff]
		return featureVector
	
if __name__ == '__main__':
	c = TestCommander("a")
	c.initialize();
