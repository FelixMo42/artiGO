int FR_M_CL = 13;
int FR_M_CCL = 7;
int FR_M_S = 9;

int FL_M_CL = 12;
int FL_M_CCL = A0;
int FL_M_S = 11;

int BR_M_CL = 8;
int BR_M_CCL = 5;
int BR_M_S = 3;

int BL_M_CL = 10;
int BL_M_CCL = 4;
int BL_M_S = 6;

int TRIG = 2; //nessisary

int BR_US = 9;
int FR_US = 8;
int FC_US = A5;
int FL_US = A4;
int BL_US = A5;

int outputs[] = {
    FR_M_CL, FR_M_CCL, FR_M_S,
    FL_M_CL, FL_M_CCL, FR_M_S,

    BR_M_CL, BR_M_CCL, BR_M_S,
    BL_M_CL, BL_M_CCL, BR_M_S,

    TRIG
};

int inputs[] = {
  BR_US, FR_US, FC_US, FL_US, BL_US
};

void setup() {
  Serial.begin(9600);

  for (int port : outputs) {
    pinMode(port, OUTPUT);
  }
  for (int port : inputs) {
    pinMode(port, INPUT);
  }
}

int dist() {
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  return pulseIn(FC_US, HIGH);
}

void set(String m, int s) {
  if (m.equals("FL")) {
    analogWrite(FL_M_S, s);

    if (s > 0) {
      digitalWrite(FL_M_CL, HIGH);
      digitalWrite(FL_M_CCL, LOW);
    } else if (s == 0) {
      digitalWrite(FL_M_CL, LOW);
      digitalWrite(FL_M_CCL, LOW);
    } else if (s < 0) {
      digitalWrite(FL_M_CL, LOW);
      digitalWrite(FL_M_CCL, HIGH);
    }
  } else if (m.equals("FR")) {
    analogWrite(FR_M_S, s);

    if (s > 0) {
      digitalWrite(FR_M_CL, HIGH);
      digitalWrite(FR_M_CCL, LOW);
    } else if (s == 0) {
      digitalWrite(FR_M_CL, LOW);
      digitalWrite(FR_M_CCL, LOW);
    } else if (s < 0) {
      digitalWrite(FR_M_CL, LOW);
      digitalWrite(FR_M_CCL, HIGH);
    }
  } else if (m.equals("BL")) {
    analogWrite(BL_M_S, s);

    if (s > 0) {
      digitalWrite(BL_M_CL, HIGH);
      digitalWrite(BL_M_CCL, LOW);
    } else if (s == 0) {
      digitalWrite(BL_M_CL, LOW);
      digitalWrite(BL_M_CCL, LOW);
    } else if (s < 0) {
      digitalWrite(BL_M_CL, LOW);
      digitalWrite(BL_M_CCL, HIGH);
    }
  } else if (m.equals("BR")) {
    analogWrite(BR_M_S, s);

    if (s > 0) {
      digitalWrite(BR_M_CL, HIGH);
      digitalWrite(BR_M_CCL, LOW);
    } else if (s == 0) {
      digitalWrite(BR_M_CL, LOW);
      digitalWrite(BR_M_CCL, LOW);
    } else if (s < 0) {
      digitalWrite(BR_M_CL, LOW);
      digitalWrite(BR_M_CCL, HIGH);
    }
  }
}

void loop() {
  int d = 0;
  for (int i = 0; i < 5; i++) {
    int n = dist();
    if (n > 100) {
      d += n;
    }
    delay(15);
  }
  d /= 5;

  Serial.println(d);

  /*if (d > 500) {
    set("FL", 100);
    set("FR", 100);
    set("BL", 100);
    set("BR", 100);
  } else {
    set("FL", 0);
    set("FR", 0);
    set("BL", 0);
    set("BR", 0);
  }*/

  delay(500);
}
