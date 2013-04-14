from pyneurgen.neuralnet import NeuralNet
import numpy as np
# Your AI for CTF must inherit from the base Commander class.  See how this is
# implemented by looking at the commander.py in the ./api/ folder.
from api import Commander

# The commander can send 'Commands' to individual bots.  These are listed and
# documented in commands.py from the ./api/ folder also.
from api import commands

# The maps for CTF are layed out along the X and Z axis in space, but can be
# effectively be considered 2D.
from api import Vector2
import random
import time
import threading

class TestCommander(Commander):
	"""Initial test commander for basic reinforcement learning"""

	teamName = ""
	goalLoc = Vector2(0,0)
	respawnTime = 0.0
	maxRespawnTime = 45.0
	moveDen = 0.0
	sightDen = 0.0
	normDistance = 0.0
	net = NeuralNet()
	fvSize = 15
	bvSize = 10
	outSize = 7
	#shows up in sandbox files on linux atleast
	fileName = "testWeightsFix.txt"
	#toggle train and test
	train = 1
	#used fo tind the max or min fitnes values for normalization
	fitMax = [1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	fitMin = np.multiply(fitMax,-1.0)
	fitMaxVal = 0
	fitMinVal = 0
	firstRun = True
	priorFvs = {}
	defRotate = [(Vector2(1,0),0.1),(Vector2(0,1),0.1),(Vector2(0,-1),0.1),(Vector2(-1,0),0.1)]
	lock = threading.Lock()
	
	def initialize(self):
		#grabs initial game state for the game
		gameState = self.game
		#grabs initial level state for the level
		lvlState = self.level
		#calculate normalization distance
		d1 = lvlState.area[0]
		d2 = lvlState.area[1]
	    	self.normDistance = d1.distance(d2)
	    	#calculate visible and moveable areas on the map
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
	     	self.moveDen = moveDensity/totalArea
	     	self.sightDen = visibleDensity/totalArea
		
	     	#neural net setup
	     	inputSize = self.bvSize + self.fvSize
	     	hiddenSize = (inputSize+self.outSize)/2
	     	#self.net.init_layers(inputSize, [hiddenSize], self.outSize)
	     	#self.net.randomize_network()
	     	#self.net.set_learnrate(.1)
	     	#self.net.save(self.fileName)

	     	#self.lock.acquire()
	     	#try:
	     	#	self.net.load(self.fileName)
		#finally:
		#		self.lock.release()
	     	self.fitMaxVal = self.fitness(self.fitMax)
	     	self.fitMinVal = self.fitness(self.fitMin)

	def tick(self):
		#global feature vector
		self.fv = list(self.getFeatureVector())
	    	#feature vectors for each bot
	    	self.bfvs = self.getBotFeatureVectors()
		#print bfvs
		#print fv
		
		for bot in self.game.bots_available:
			#output from neural net determines bot action (note: other actions aren't tested yet)
			comboFv = list(self.fv + self.bfvs[bot.name])
			comboFv = [float(i) for i in comboFv]
			botActionChoice = random.randint(0,6);

		#	if self.train == 1 and self.firstRun == False:
				#Update Neural Net by feeding last command bot did and updating error to match fitness change
		#		lastBotCommand = self.priorFvs[bot.name]
		#		botActionChoice = random.randint(0,6);
		#		delta = np.subtract(lastBotCommand[1],comboFv)
		#		error = self.fitness(delta)
		#		norm_error = (error - self.fitMinVal)/(self.fitMaxVal - self.fitMinVal)
		#		self.net.input_layer.load_inputs(comboFv)
		#		self.net._feed_forward()
		#		activations = self.net.output_layer.activations()
		#		target = activations
		#		target[lastBotCommand[0]] = norm_error
		#	elif self.train == 1:
		#		botActionChoice = random.randint(0,6);
		#	else:
		#		pass
		#		self.net.input_layer.load_inputs(comboFv)
		#		self.net._feed_forward()
		#		activations = self.net.output_layer.activations()
		#		botActionChoice = activations.index(max(activations))
		#	self.priorFvs[bot.name] = [botActionChoice, comboFv]
			
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
		 			self.issue(commands.Charge, bot, enBotPos, description = 'Charge nearest enemy')

			elif botActionChoice == 3:       
                              #defend current position
			      self.issue(commands.Defend, bot, self.defRotate, description = 'Defend position')

			elif botActionChoice == 4:
                  		#defend flag
			      flagPosition = self.game.team.flag.position
			      if bot.position.distance(flagPosition) > 2:
				      self.issue(commands.Charge, bot, flagPosition, description = 'Move to flag location')
			      else:
				      self.issue(commands.Defend, bot, self.defRotate,description = 'Defend the flag')
			elif botActionChoice == 5:
	                      #attack enemy flag carrier
			      flagPosition = self.game.team.flag.position
			      if self.game.team.flag.carrier != None:
				      self.issue(commands.Charge, bot, flagPosition, description = 'Attack Flag Carrier')
		      	elif botActionChoice == 6:
			      #camp enemy spawn point
			      enSpawn = self.game.enemyTeam.botSpawnArea
			      enSpawn = enSpawn[0].midPoint(enSpawn[1])
			      if bot.position.distance(enSpawn) > 20:
				      self.issue(commands.Charge, bot, enSpawn, description = 'Run to enemy spawn')
			      elif bot.position.distance(enSpawn) < 2:
				      self.issue(commands.Defend, bot, self.defRotate, description = 'Defend at enemy spawn')
			      else: 
				      self.issue(commands.Attack, bot, enSpawn, enSpawn, description = 'Attack Enemy Spawn')
		self.firstRun = False
	     
	def shutdown(self):
		pass
	#	if self.train == 1:
	#		self.lock.acquire()
	#		try:
	#			self.net.save(self.fileName)
	#		finally:
	#			self.lock.release()
		
	def fitness(self, comboFv):
		fit = comboFv[0]*0.3        #flagCarried, 
		fit = fit - 0.3*comboFv[1]  #ourFlagCarried,
		fit = fit + 0.1*comboFv[2] #myFlagDist, 
		fit = fit - 0.1*comboFv[3] #theirFlagDist,
        #respawnTime, 4
		#timeToRespawn, 5 
		#timeLeft, 6
        #totalBotNum,7
		fit = fit + 0.2*comboFv[8]#totalBots,  8
		fit = fit - 0.2*comboFv[9] #enemyNumber,
		fit = fit + 0.4*comboFv[10] #scoreDiff,
        #moveDensity, 11
        #visibleDensity,12
        #ourFlagDespawn, 13
        #enemyFlagDespawn 14
		fit = fit + 0.5*comboFv[15] #hasFlag,
		fit = fit + 0.01*comboFv[16]  #isAlive, 16
		fit = fit + 0.001*comboFv[17] #seesEnemy, 17
		fit = fit - 0.0005*comboFv[18] #seenByEnemy, 18
		fit = fit + 0.001*comboFv[19] #distToEnemyFC, 19 
		fit = fit + 0.001*comboFv[20] #distToFriendlyFC,20
		fit = fit + 0.001*comboFv[21] #distToNearestEnemy,21 
		fit = fit + 0.001*comboFv[22] #distToNearestAlly, 22
        #distToFriendlySpawn,23
        #distToEnemySpawn 24
		return fit
	
	
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
		flagCarried = 0.0
		enFlag = enemyInfo.flag.name
		if(enemyInfo.flag.carrier != None):
			flagCarried = 1.0
		#is our flag taken
		ourFlagCarried = 0.0
		ourFlag = myInfo.flag.name
		if(myInfo.flag.carrier != None):
			ourFlagCarried = 1.0
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
		if enemyScore == 0 & myScore == 0:
			scoreDiff = 0.5;
		else:
			scoreDiff = myScore*1.0/(myScore + enemyScore)

                #density of moveable and visible area
                moveDensity = self.moveDen
                visibleDensity = self.sightDen
                #time until friendly flag despawns
		if myInfo.flag.respawnTimer < 0:
                        ourFlagDespawn = 0
                else:
                        ourFlagDespawn = myInfo.flag.respawnTimer/30.0
		#time until enemy flag despawns
		if enemyInfo.flag.respawnTimer < 0:
                        enemyFlagDespawn = 0.0
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
		normDist = self.normDistance

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
                        hasFlag = 0.0
                        if bot.flag != None:
                                hasFlag = 1.0
                        #Health
                        isAlive = 0.0
                        if bot.health > 0:
                                isAlive = 1.0
                        #See enemy
                        seesEnemy = 0.0
                        if len(bot.visibleEnemies) > 0:
                                seesEnemy = 1.0
                        #Seen by enemy
                        seenByEnemy = 0.0
                        if len(bot.seenBy) > 0:
                                seenByEnemy = 1.0
                        #distance to enemy FC
                        distToEnemyFC = 0.0
                        if myInfo.flag.carrier != None:
                                distToEnemyFC = botPos.distance(myInfo.flag.carrier.position)/normDist
                        #distance to our flag/FC
                        distToFriendlyFC = 0.0
                        if enemyInfo.flag.carrier != None:
                                distToFriendlyFC = botPos.distance(enemyInfo.flag.carrier.position)/normDist
                        #distance to nearest seen enemy
                        distToNearestEnemy = 1.0
                        if len(bot.visibleEnemies) > 0:
                                for enBot in bot.visibleEnemies:
                                        tDist = botPos.distance(enBot.position)/normDist
                                        if tDist < distToNearestEnemy:
                                                distToNearestEnemy = tDist
                        #distance to nearest ally
                        distToNearestAlly = 1.0
                        for frBot in myInfo.members:
                               if frBot != bot:
                                       tDist = botPos.distance(frBot.position)/normDist
                                       if tDist < distToNearestAlly:
                                               distToNearestAlly = tDist


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
               
