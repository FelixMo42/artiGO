int RM_CL = 10;
int RM_CCL = 11;
int RM_S = 6;

int LM_CL = 8;
int LM_CCL = 9;
int LM_S = 5;

int ECHO = 3;
int TRIG = 13;

int LED = 13;

int outputs[] = {
    RM_CL, RM_CCL, RM_S,
    LM_CL, LM_CCL, LM_S,
    TRIG,
    LED
};

int inputs[] = {
  ECHO
};

void setup() {
  Serial.begin(9600);
  
  for (int port : outputs) {
    pinMode(port, OUTPUT);
  }
  for (int port : inputs) {
    pinMode(port, INPUT);
  }

  digitalWrite(RM_CL, HIGH);
}

int dist() {
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(100);
  digitalWrite(TRIG, LOW);
  return pulseIn(ECHO, HIGH);
}

void loop() {
  int d = dist();
  //Serial.println(d);
  //delay(500);
  if (d > 100) {
    analogWrite(RM_S, 100);
  } else {
    analogWrite(RM_S, 0);
  } 
}
