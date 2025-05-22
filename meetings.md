# Meeting 22.05

At this point, first shot at building a PINN for the poiseuille flow <br>
The simulation area looked like this: <br>
![img](/images/domain1.png) <br>
and after training the PINN for a while, the result of the simulation was the following: <br>
![img](/images/simu1.png)<br>

After the meeting a few points have been talked about:
- We need to have more information on some part of the domain, but this might not be the best way to do it (corners aren't the most important part)
- We would like to train on a larger domain than what we want to display/analyse, not like what we were doing here staying only on the domain
- We need to also display pressure in the results graph
- We need to display the Loss, for both the physics and BCs parts on the same graph
- We need to find and solve the analytical version for the poiseuille flow in order to have a baseline to compare to
- We have to check the different terms for the Navier-Stokes equation and adjust their weight if necessary