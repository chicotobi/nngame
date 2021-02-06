float t = 0; //current time, t
float m = 1; //pendulum mass, kg
float M = 1; //cart mass, kg
float l = 2; //pendulum length, meters
float x = 0; //cart position, meters
float v = 0; //cart velocity, meters/s
float a = 0; //cart acceleration, meters/s^2
float theta = 0.0; //pendulum angle from vertical
float omega = 0; //pendulum angular velocity
float alpha = 0; //pendulum angular acceleration

//constants
float g = 9.81; //gravitational acceleration, m/s^2
float dt = 0.01; //TODO: use framerate

//purely for display
float cartwidth = 0.2;
float cartheight = 0.2;
float pendsize = 0.1;

boolean control=true;

float x_goal = 1.0;

void reset(){
  x=v=a=theta=omega=alpha=0;
}

float getAlpha(float F){
  //kg*m terms
  float t1 = (M+m)*l/cos(theta);
  float t2 = -m*l*cos(theta);
  
  //force terms
  float f1 = (M+m)*g*sin(theta)/cos(theta);
  float f2 = -m*l*sq(omega)*sin(theta);
  
  float alpha = (F+f1+f2)/(t1+t2);
  return alpha;
}

float getAcc(float alpha){
  return (l*alpha - g*sin(theta))/cos(theta);
}

void setAccelerations(float F){
  alpha = getAlpha(F);
  a = getAcc(alpha);
}

void updateState(){
  t += dt;
  
  omega += dt*alpha;
  theta += dt*omega;
  v += dt*a;
  x += dt*v;
}

float forceForAngularAcceleration(float alpha){
  float t1 = (M+m)*l/cos(theta);
  float t2 = -m*l*cos(theta);
  float f1 = -(M+m)*g*sin(theta)/cos(theta);
  float f2 = m*l*sq(omega)*sin(theta);
  
  float F = (t1+t2)*alpha + f1 + f2;
  return F;
}

void setup(){
  size(1000,400);
  strokeWeight(0.01);
}

int sign(float x){
  if(x<0) return -1;
  else return 1;
}

void draw(){
  background(255);
  
  float F = 0;
  float F_control = 0;
  if(mousePressed){
    F = (mouseX-width/2)*0.1;
    line(width/2, mouseY, mouseX, mouseY);
  } else {
    if(control)
      F_control = -theta*100 - omega*50 + v*10 + (x-x_goal)*4.4;
      F_control = max(-1,min(1,F_control));
      //F_control = -theta*100 - omega*50 + v*10 + (x-x_goal)*3;
  }
 
  setAccelerations(F+F_control);
  updateState();
  
  //show control force
  stroke(255,0,0);
  strokeWeight(1.0);
  line(width/2, 3*height/4, width/2+F_control*10, 3*height/4);
  line(width/2, 3*height/4+10, width/2+F_control*100, 3*height/4+10);
  
  strokeWeight(0.01);
  stroke(0);
  
  fill(0);
  text(String.format("%.02f",t)+" s",10,20);
  if(control)
    text("stability [c]ontrol ON",10,35);
  else
    text("stability [c]ontrol OFF",10,35);
  
  translate(width/2,2*height/3);
  scale(100,-100);
  
  line(x_goal,0,x_goal,-0.3);
  
  noFill();
  rect(x-cartwidth/2,-cartheight/2,cartwidth,cartheight);
  
  float pendX = x - sin(theta)*l;
  float pendY = cos(theta)*l;
  line(x,0,pendX,pendY);
  
  pushMatrix();
  translate(pendX, pendY);
  rotate(theta);
  rect(0-pendsize/2,0-pendsize/2,pendsize,pendsize);
  popMatrix();
  

}

void keyPressed(){
  if(key==' '){
    reset();
  } 
  if(key=='c'){
    control = !control;
  }
  if(keyCode==LEFT){
    x_goal -= 0.1;
  }
  if(keyCode==RIGHT){
    x_goal += 0.1;
  }
}
