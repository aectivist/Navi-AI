import SettingsAPI_Disc
import random
import discord
from discord.ext import commands
import ollama
#ollama 
url = "http://localhost:11434/api/generate"

logger = SettingsAPI_Disc.logging.getLogger("bot")
def run():
    intents = discord.Intents.default()
    intents.message_content = True  # Enable message content intent if needed
    intents.members = True # Enable members intent if needed
    

    bot = commands.Bot(command_prefix='!', intents=intents)

    @bot.event
    async def on_ready():
        logger.info(f'Logged in as {bot.user} (ID: {bot.user.id})')
        print("____________")
    
    @bot.command()
    async def ping(ctx):
        #await ctx.message.author.send("Hello")
        discord.utils.get()


    @bot.command()
    async def input(ctx, *what):
        input = " ".join(what)
        res = ollama.generate(model="NAVI", prompt=input)
        output = str(res["response"])
        await ctx.send(output)

    bot.run(SettingsAPI_Disc.DISCORDAPI, root_logger=True)  

try:
    if __name__=="__main__":
        run()
except Exception as e: 
    print(f"Error in Discord Bot: {e}")

#async def ping(ctx):
#        """Answers with pong"""
#        await ctx.send('Pong!')
