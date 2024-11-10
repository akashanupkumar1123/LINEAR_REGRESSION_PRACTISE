***You just got some contract work with an Ecommerce company based in New York City that sells clothing online but they also have in-store style        and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order        either on a mobile app or website for the clothes they want.

     The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired you on contract to      help them figure it out! Let's get started!



*** Get the Data
We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:

Avg. Session Length: Average session of in-store style advice sessions.
Time on App: Average time spent on App in minutes
Time on Website: Average time spent on Website in minutes
Length of Membership: How many years the customer has been a member.
** Read in the Ecommerce Customers csv file as a DataFrame called customers.**

customers = pd.read_csv("Ecommerce Customers")




***Let's explore the data!

->For the rest of the exercise we'll only be using the numerical data of the csv file.

->Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns



->Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.




->Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?


->Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets. ** Set a variable X equal to the numerical     features of the customers and a variable y equal to the "Yearly Amount Spent" column. **


->Now that we have fit our model, let's evaluate its performance by predicting off the test values!

** Use lm.predict() to predict off the X_test set of the data.**

***Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).

** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.


***You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data.

***Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().


We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important. Let's see if we can interpret the coefficients at all to get an idea.


***Interpreting the coefficients:

Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.


This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better. This sort of answer really depends on the other factors going on at the company, you would probably want to explore the relationship between Length of Membership and the App or the Website before coming to a conclusion!

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------







