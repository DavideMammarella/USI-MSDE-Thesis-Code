<p align=center>
	<img src="https://upload.wikimedia.org/wikipedia/commons/c/cd/USI_Logo.svg" alt="Logo USI" width="132" style="margin: 0px 15px 0px 0px;"/>
    <img src="https://shd.unimib.it/wp-content/uploads/sites/3/2018/02/unimib_logo-istituzionale_vettoriale-copy.png" alt="Logo UniMiB" width="132" style="margin: 0px 0px 0px 15px;"/>
</p>
<h1 align=center>Predicting Safety-Critical Misbehaviours in Autonomous Driving Systems using Uncertainty Quantification</h1>
<p align=center>
	Master's thesis for the double degree master program in <br>
    Software and Data Engineering at<br> 
    Università della Svizzera italiana (Lugano, Switzerland) and <br>
    Università degli Studi di Milano-Bicocca (Milan, Italy).
</p>


<!-- ABSTRACT -->

## Abstract

Deep Neural Network (DNN) based autonomous driving systems are implemented using finite - hence limited - training sets that hardly represent all the conditions that can be met at testing time. Thus, DNN supervisors analyze whether the autopilot is confident about its predictions, prior to sending them to the car's actuators.

The goal of the thesis is to design a predictive model for failure prediction in autonomous driving systems based on white-box confidence measures, such as uncertainty quantification. The approach will use information internal to the autopilot to predict incoming failures in a self-driving car. We will evaluate the approach at predicting driving requirements violations such as collisions or wavy/erratic driving in a driving simulator.

In summary, the questions the thesis answer are: Can we build an accurate misbehavior predictor for a DNN driving model based on white-box metrics? How does it compare to existing black-box approaches? The results of the thesis increase the degree of support available to engineers in self-driving cars development and testing. 

**Keywords**: Software Engineering; Deep Learning; Anomaly Detection; Software Testing; Deep Learning Testing; White-Box Testing; Autonomous Vehicles; Self-Driving Cars; 

<!-- DOCUMENTATION -->

## Documentation

Documentation regarding the installation, usage and replication of experiments can be found in [`docs`](/docs).

<!-- LICENSE -->

## License

Distributed under the MIT License. <br>
See [`LICENSE`](LICENSE) for more information.
