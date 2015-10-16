// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

//Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/constants.hpp>

// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <queue>
#include <stack>
#include <limits>

#include <../cubespheres/helper.h>

#define NUMSPHERES 800

using namespace glm;

class Sphere {
public:
	GLUquadric *body;
	float t0;
	vec3 q0;
	vec3 velocity;
	vec3 color;
	int index, cols;
	bool empty;

	float nextCollision;
	vec3 newVelocity;

	Sphere(){}
	Sphere(vec3 q, vec3 velocityV, float t, int ind, vec3 colort ){
		q0 = q;
		velocity = velocityV;
		t0 = t;
		index = ind;
		body = gluNewQuadric();
		cols = 0;
		empty = false;
		color = colort;
	}
	
	void setAttributes(vec3 q, vec3 velocityV, float t, int ind, vec3 colort){
		q0 = q;
		velocity = velocityV;
		t0 = t;
		index = ind;
		body = gluNewQuadric();
		cols = 0;
		empty = false;
		color = colort;
	}

};

class CollisionEvent {
public:
	int i,j;
	float timeOccuring;
	int icols, jcols;

	CollisionEvent(){}
	CollisionEvent (int sphereI, int sphereJ, float nextCollision, float inserted, int cols, int cols2){
		i = sphereI;
		j = sphereJ;
		timeOccuring = nextCollision;
		icols = cols;
		jcols = cols2;
	}
};

class CompareCollisionTimes {
public:
	bool operator() (CollisionEvent s1, CollisionEvent s2)
	{
		if (s1.timeOccuring > s2.timeOccuring)
			return true;
		return false;
	}
};

std::priority_queue<CollisionEvent, std::vector<CollisionEvent>, CompareCollisionTimes> events;
std::vector<Sphere>::iterator sphereIterInit;
std::vector<Sphere> sphere;
std::stack<int> freepos;

int fullscreen = 0, numSpheres = 0, showtraj = 0, light=0, dex=0, texture=0, shading=0;
float velocity, radius, elapsedTime, frontback=0.0f, updown=0.0f, leftright=0.0f;
vec3 position;

float frameTime, newTime, currentTime, accumulator=0.0f, dt=0.01f, alpha;

GLsizei		width  = 1024;
GLsizei		height = 768;

GLfloat ambientA[]= { 0.5f, 0.5f, 0.5f, 0.5f };
GLfloat diffuse[]= { 1.0f, 1.0f, 1.0f, 1.0f, 0.0f }; 

//GLfloat specular[]= { 0.5f, 0.5f, 0.5f };
//GLfloat shininess[] = { 5.0f };

GLfloat Position[][10] = 
{ 
	{0.0f, 0.0f, 0.0f, 1.0f},
	{0.9f, 0.9f, 0.9f, 1.0f},
	{-0.9f, 0.9f, 0.9f, 1.0f},
	{-0.9f, 0.9f, -0.9f, 1.0f},
	{0.9f, 0.9f, -0.9f, 1.0f},
	{0.9f, -0.9f, 0.9f, 1.0f},
	{-0.9f, -0.9f, 0.9f, 1.0f},
	{-0.9f, -0.9f, -0.9f, 1.0f},
	{0.9f, -0.9f, -0.9f, 1.0f},
	{0.0f, -0.9f, 0.0f, 1.0f}
};
GLfloat PositionD[][10] = 
{ 
	{0.0f, 0.9f, 0.0f, 0.0f},
	{0.9f, 0.9f, 0.9f, 0.0f},
	{-0.9f, 0.9f, 0.9f, 0.0f},
	{-0.9f, 0.9f, -0.9f, 0.0f},
	{0.9f, 0.9f, -0.9f, 0.0f},
	{0.9f, -0.9f, 0.9f, 0.0f},
	{-0.9f, -0.9f, 0.9f, 0.0f},
	{-0.9f, -0.9f, -0.9f, 0.0f},
	{0.9f, -0.9f, -0.9f, 0.0f},
	{0.0f, -0.9f, 0.0f, 0.0f}
};
GLfloat spotDirection[][10] = 
{ 
	{0.0f, -1.0f, 0.0f, 0.0f},
	{-1.0f, -1.0f, -1.0f, 0.0f},
	{1.0f, -1.0f, -1.0f, 0.0f},
	{1.0f, -1.0f, 1.0f, 0.0f},
	{-1.0f, -1.0f, 1.0f, 0.0f},
	{-1.0f, 1.0f, -1.0f, 0.0f},
	{1.0f, 1.0f, -1.0f, 0.0f},
	{1.0f, 1.0f, 1.0f, 0.0f},
	{-1.0f, 1.0f, 1.0f, 0.0f},
	{0.0f, 1.0f, 0.0f, 0.0f}
};
GLfloat cut_off[] = { 25.0f };

GLuint ballTexture, wallTexture, floorTexture,
		topSide, bottomSide, leftSide, backSide, rightSide;

//COLLISION FUNCTIONS
void calculateWallCollisionFor(int i)
{
	float tx=1000.0f, ty=1000.0f, tz=1000.0f;
	float min = 1000.0f;

	sphere[i].newVelocity = vec3(1.0f, 1.0f, 1.0f);

	vec3 p = sphere[i].q0;
		
	//Check collision with walls
	if ( sphere[i].velocity.x > 0.0f )
	{
		tx = (1.0f-radius-p.x)/sphere[i].velocity.x;
		min = (tx<min)?tx:min;
	}
	else if ( sphere[i].velocity.x < 0.0f )
	{
		tx = -(1.0f-radius+p.x)/sphere[i].velocity.x;
		min = (tx<min)?tx:min;
	}
	if ( sphere[i].velocity.y > 0.0f )
	{
		ty = (1.0f-radius-p.y)/sphere[i].velocity.y;
		min = (ty<min)?ty:min;
	}
	else if ( sphere[i].velocity.y < 0.0f )
	{
		ty = -(1.0f-radius+p.y)/sphere[i].velocity.y;
		min = (ty<min)?ty:min;
	}
	if ( sphere[i].velocity.z > 0.0f )
	{
		tz = (1.0f-radius-p.z)/sphere[i].velocity.z;
		min = (tz<min)?tz:min;
	}
	else if ( sphere[i].velocity.z < 0.0f )
	{
		tz = -(1.0f-radius+p.z)/sphere[i].velocity.z;
		min = (tz<min)?tz:min;
	}
	//-----------------------------------------------------------------
	if (tx==min){
		sphere[i].newVelocity *= vec3(-1.0f,1.0f,1.0f);
	}
	if (ty==min){
		sphere[i].newVelocity *= vec3(1.0f,-1.0f,1.0f);
	}
	if (tz==min){
		sphere[i].newVelocity *= vec3(1.0f,1.0f,-1.0f);
	}
	sphere[i].nextCollision = min + elapsedTime;
}
int calculateBallCollisionFor(int i){
	int j=0, balltoball=-1;
	vec3 q, v, factor, p1, p2;
	float a, b, c, D, t1, t2, t, min;

	min = sphere[i].nextCollision;
	if (i==j)j++;
	while(j<numSpheres){
		if(j!=i){
			p1 = sphere[i].q0 + (elapsedTime - sphere[i].t0)*sphere[i].velocity;
			p2 = sphere[j].q0 + (elapsedTime - sphere[j].t0)*sphere[j].velocity;
			
			q = p1 - p2;
			v = sphere[i].velocity - sphere[j].velocity;

			a = dot(v, v);
			b = 2*dot(q, v);
			c = dot(q, q) - 4*radius*radius;

			D = b*b - 4*a*c;

			if ( D >= 0 ) {//b>0?
				t1 = (-b + sqrt(D))/(2*a);
				t2 = (-b - sqrt(D))/(2*a);
					
				t = (t2<t1)?t2:t1;
				
				if( t < min && t>0 ){
					min = t;
					balltoball = j;	
				}
			}
		}
		j++;
	}
	j = balltoball;
	if (j!=-1){
		sphere[i].nextCollision = min + elapsedTime;
		sphere[j].nextCollision = min + elapsedTime;
	}
	
	return balltoball;
}
void calculateCollision(int i){
	CollisionEvent col1;
	int balltoball = -1;

	calculateWallCollisionFor(i);
	col1 = CollisionEvent(i, -1, sphere[i].nextCollision, elapsedTime, sphere[i].cols, 0);
	events.push(col1);

	balltoball = calculateBallCollisionFor(i);

	if (balltoball!=-1){
		col1 = CollisionEvent(i, balltoball, sphere[i].nextCollision, elapsedTime,  sphere[i].cols, sphere[balltoball].cols);
		events.push(col1);
	}
}
int checkOutdated(CollisionEvent event){
	int i, j;

	i = event.i;
	j = event.j;

	if (sphere[i].empty)
		return 1;

	if( j==-1 ){ //only one ball
		if (sphere[i].cols == event.icols)
			return 0;
	}else{		//two balls
		if (sphere[j].empty)
			return 1;
		if (sphere[i].cols == event.icols && sphere[j].cols == event.jcols)
			return 0;
	}
	return 1;
}
void checkCollision(){
	int i, j;
	vec3 p1, p2, q, v, factor;
	while( !events.empty() && elapsedTime >= events.top().timeOccuring)
	{
		i = events.top().i;

		if( checkOutdated( events.top() ) ){		//check for invalid collision
			events.pop();
			//j = events.top().j;
			//if( j != -1 ){
			//	printf("Popped a ball-ball between %d and %d\n", i, j );
			//}
		} else if (events.top().j!=-1) {	//ball collision
			j = events.top().j;

			sphere[i].q0 += (events.top().timeOccuring - sphere[i].t0) * sphere[i].velocity;
			sphere[j].q0 += (events.top().timeOccuring - sphere[j].t0) * sphere[j].velocity;

			q = sphere[i].q0 - sphere[j].q0;
			v = sphere[i].velocity - sphere[j].velocity;

			factor = q*dot(v,q)/(4*radius*radius);

			sphere[i].velocity -= factor;
			sphere[i].t0 = events.top().timeOccuring;

			sphere[j].velocity += factor;
			sphere[j].t0 = events.top().timeOccuring;

			sphere[i].cols++;
			sphere[j].cols++;

			events.pop();
			calculateCollision(i);
			calculateCollision(j);
		}else{//wall collision
			sphere[i].q0 += (events.top().timeOccuring - sphere[i].t0) * sphere[i].velocity;
			sphere[i].velocity *= sphere[i].newVelocity;
			sphere[i].t0 = events.top().timeOccuring;

			sphere[i].cols++;

			events.pop();
			calculateCollision(i);
		}
	}
}

// Ball creation and deletion
void createSphere(vec3 startPos, vec3 velocity, float t0, vec3 color = vec3(0.6f, 0.4f, 0.4f));
void createSphere(vec3 startPos, vec3 velocity, float t0, vec3 color){
	int id;
	if(sphere.size()<NUMSPHERES){
		if(!freepos.empty()){
			id = freepos.top();
			freepos.pop();
			sphere[id].setAttributes(startPos, velocity, t0, id, color);
			sphereIterInit = sphereIterInit - 1;
		}else{
			id = numSpheres;
			numSpheres++;
			sphere.push_back(Sphere(startPos, velocity, t0, id, color));
			sphereIterInit = sphere.begin();
		}
		calculateCollision(id);
	}
	printf("Number of balls %d\n", numSpheres);
}
void destroySphere(){
	int i = sphereIterInit->index;
	if(sphereIterInit != sphere.end()-1){
		freepos.push(i);
		sphere[i].empty = true;
		sphereIterInit = sphereIterInit + 1;
	}else
		printf("Hey if you pull them all out this sim has no point!\n");
}

//Initialize OpenGL
int InitGL(GLvoid)
{
	float ix, iy, iz, step;

	GLfloat mat_shininess[] = { 50.0 };
	if (height==0)										// Prevent A Divide By Zero By
	{
		height=1;										// Making Height Equal One
	}

	glViewport(0,0,width,height);						// Reset The Current Viewport

	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix

	// Calculate The Aspect Ratio Of The Window
	gluPerspective(65.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();

	glShadeModel(GL_SMOOTH);
	glEnable(GL_COLOR_MATERIAL);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	// Create the box using display lists----------------------------------------
	step = 0.01f;
	topSide = glGenLists(1);
	glNewList(topSide, GL_COMPILE);

	glColor3f(0.8f,0.8f,0.2f);
	//glColor3f(1.0f,1.0f,1.0f);
	iy=1.0f;
	for(ix=1.0f; ix>-1.0f; ix-=step){
		for(iz=1.0f; iz>-1.0f; iz-=step){
			glBegin(GL_TRIANGLES);
				glNormal3f(0.0f, -1.0f, 0.0f);
				glVertex3f( ix,		iy,	iz);
				glVertex3f( ix-step,iy,	iz-step);
				glVertex3f( ix-step,iy,	iz); 
				
				glNormal3f(0.0f, -1.0f, 0.0f);
				glVertex3f( ix,		iy,	iz);
				glVertex3f( ix,		iy, iz-step);
				glVertex3f( ix-step,iy,	iz-step);
			glEnd();
		}
	}
	glEndList();

	bottomSide = glGenLists(1);
	glNewList(bottomSide, GL_COMPILE);

	glColor3f(1.0f,1.0f,1.0f);
	iy=-1.0f;
	for(ix=1.0f; ix>-1.0f; ix-=step){
		for(iz=1.0f; iz>-1.0f; iz-=step){
			glBegin(GL_TRIANGLES);
				glNormal3f(0.0f, 1.0f, 0.0f);
				glVertex3f( ix,		iy,	iz);
				glVertex3f( ix-step,iy,	iz-step);
				glVertex3f( ix-step,iy,	iz); 
				
				glNormal3f(0.0f, 1.0f, 0.0f);
				glVertex3f( ix,		iy,	iz);
				glVertex3f( ix,		iy, iz-step);
				glVertex3f( ix-step,iy,	iz-step);
			glEnd();
		}
	}
	glEndList();

	rightSide = glGenLists(1);
	glNewList(rightSide, GL_COMPILE);

	glColor3f(1.0f,0.0f,0.0f);
	//glColor3f(1.0f,1.0f,1.0f);
	ix=1.0f;
	for(iy=1.0f; iy>-1.0f; iy-=step){
		for(iz=1.0f; iz>-1.0f; iz-=step){
			glBegin(GL_TRIANGLES);
				glNormal3f(-1.0f, 0.0f, 0.0f);
				glVertex3f( ix,		iy,		iz);
				glVertex3f( ix,	iy-step,	iz-step);
				glVertex3f( ix,	iy-step,	iz); 
				
				glNormal3f(-1.0f, 0.0f, 0.0f);
				glVertex3f( ix,	iy,		iz);
				glVertex3f( ix,	iy,		iz-step);
				glVertex3f( ix,	iy-step,iz-step);
			glEnd();
		}
	}
	glEndList();

	leftSide = glGenLists(1);
	glNewList(leftSide, GL_COMPILE);

	glColor3f(0.0f,0.0f,1.0f);
	//glColor3f(1.0f,1.0f,1.0f);
	ix=-1.0f;
	for(iy=1.0f; iy>-1.0f; iy-=step){
		for(iz=1.0f; iz>-1.0f; iz-=step){
			glBegin(GL_TRIANGLES);
				glNormal3f(1.0f, 0.0f, 0.0f);
				glVertex3f( ix,		iy,		iz);
				glVertex3f( ix,	iy-step,	iz-step);
				glVertex3f( ix,	iy-step,	iz); 
				
				glNormal3f(1.0f, 0.0f, 0.0f);
				glVertex3f( ix,	iy,		iz);
				glVertex3f( ix,	iy,		iz-step);
				glVertex3f( ix,	iy-step,iz-step);
			glEnd();
		}
	}
	glEndList();

	backSide = glGenLists(1);
	glNewList(backSide, GL_COMPILE);

	glColor3f(0.0f,1.0f,0.0f);
	//glColor3f(1.0f,1.0f,1.0f);
	iz=1.0f;
	for(iy=1.0f; iy>-1.0f; iy-=step){
		for(ix=1.0f; ix>-1.0f; ix-=step){
			glBegin(GL_TRIANGLES);
				glNormal3f(0.0f, 0.0f, -1.0f);
				glVertex3f( ix,		iy,		iz);
				glVertex3f( ix-step,iy-step,iz);
				glVertex3f( ix,		iy-step,iz); 
				
				glNormal3f(0.0f, 0.0f, -1.0f);
				glVertex3f( ix,		iy,		iz);
				glVertex3f( ix-step,iy,		iz);
				glVertex3f( ix-step,iy-step,iz);
			glEnd();
		}
	}
	glEndList();
	//---------------------------------------END OF BOX CREATION----------------------
	
	return true;
}

void checkOverlap(void){
	std::vector<Sphere>::iterator it = sphereIterInit;
	std::vector<Sphere>::iterator itj;
	float dist;

	for(it; it != sphere.end(); it++){
		for(itj=it+1; itj != sphere.end(); itj++){
			dist = distance(it->q0, itj->q0);
			if(dist<2*radius-0.1)
				printf("OVERLAP %f of spheres %d and %d\n", dist, it->index, itj->index);
		}
	}
}

//Drawing
int DrawGLScene(GLvoid)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientA);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, PositionD[dex]);

	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT1, GL_POSITION, Position[dex]);
	//glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
	//glLightfv(GL_LIGHT1, GL_EMISSION, shininess);
	glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 1.0f);
	
	glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT2, GL_POSITION, Position[dex]);
	glLightfv(GL_LIGHT2, GL_SPOT_CUTOFF, cut_off );
	glLightfv(GL_LIGHT2, GL_SPOT_DIRECTION, spotDirection[dex]);
	//glLightf(GL_LIGHT2, GL_SPOT_EXPONENT, 5.0f);
	//glLightfv(GL_LIGHT2, GL_SPECULAR, specular);

	if (shading==0)
		glEnable(GL_LIGHTING);
	else
		glDisable(GL_LIGHTING);

	switch (light){
		case 0:
			glDisable(GL_LIGHT0);
			glDisable(GL_LIGHT1);
			glDisable(GL_LIGHT2);
			break;
		case 1:
			glEnable(GL_LIGHT0);
			glDisable(GL_LIGHT1);
			glDisable(GL_LIGHT2);
			break;
		case 2:
			glEnable(GL_LIGHT1);
			glDisable(GL_LIGHT0);
			glDisable(GL_LIGHT2);
			break;
		case 3:
			glEnable(GL_LIGHT2);
			glDisable(GL_LIGHT0);
			glDisable(GL_LIGHT1);
			break;
		}

	//glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
	//glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);

	glLoadIdentity();
	gluLookAt(0.0f,0.0f,-3.0f+frontback,0.0f+leftright,0.0f+updown,0.0f,0.0f,1.0f,0.0f);

	glPushMatrix();
	glCallList(topSide);
	glPopMatrix();

	glPushMatrix();
	glCallList(bottomSide);
	glPopMatrix();

	glPushMatrix();
	glCallList(rightSide);
	glPopMatrix();

	glPushMatrix();
	glCallList(leftSide);
	glPopMatrix();

	glPushMatrix();
	glCallList(backSide);
	glPopMatrix();

	//drawSpheres
	std::vector<Sphere>::iterator it = sphereIterInit;
	for (it; it != sphere.end(); it++){
		if (texture)
			glColor3f(it->color.x, it->color.y, it->color.z);
		else{
			glColor3f(1.0f, 1.0f, 1.0f);
			gluQuadricNormals(it->body, GLU_SMOOTH);
			gluQuadricTexture(it->body, GL_TRUE);
			glEnable(GL_TEXTURE_2D);
			glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
			gluQuadricTexture(it->body, GL_TRUE);
			glBindTexture(GL_TEXTURE_2D, ballTexture);
		}
		
		position = it->q0 + (elapsedTime - it->t0) * it->velocity;
		glPushMatrix();
		glTranslatef( position.x, position.y ,position.z );
		gluSphere(it->body, radius, 20, 20);
		
		glDisable(GL_TEXTURE_2D);
		
		glPopMatrix();		
	}

	//draw Lines of velocities using keyboard key T
	it = sphereIterInit;
	if(showtraj){
		for (it; it != sphere.end(); it++){
			glBegin(GL_LINES);
				glColor3f(0.5f,0.5f,0.5f);
				glVertex3f(it->q0.x, it->q0.y, it->q0.z);
				glVertex3f(it->q0.x + 2*it->velocity.x,
						   it->q0.y + 2*it->velocity.y,
						   it->q0.z + 2*it->velocity.z);
			glEnd();
		}
	}

	checkOverlap();

	return true;
}

//Handle full screen
int toggleFullscreen(GLFWwindow* window, int a)
{
    //glfwDestroyWindow(window); 

	//if( !glfwCreateWindow( width, height, "Test window", (a==0)?NULL:glfwGetPrimaryMonitor(), NULL))
	//{
	//	exit(EXIT_FAILURE);
	//}

	return 1;
}
//Define an error callback  
static void error_callback(int error, const char* description)  
{  
    fputs(description, stderr);  
    _fgetchar();  
}  
//Define the key input callback  
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)  
{  
    if ((key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) && action == GLFW_PRESS)  
		glfwSetWindowShouldClose(window, GL_TRUE);  

	// Toggle fullscreen
	if (glfwGetKey( window, GLFW_KEY_F11 ) == GLFW_PRESS )
		toggleFullscreen(window, ++fullscreen%2);

	if (key == GLFW_KEY_E && (action == GLFW_REPEAT || action == GLFW_PRESS))
		updown+=0.1f;

	if (key == GLFW_KEY_D && (action == GLFW_REPEAT || action == GLFW_PRESS))
		updown-=0.1f;

	if (key == GLFW_KEY_S && (action == GLFW_REPEAT || action == GLFW_PRESS))
		leftright+=0.1f;

	if (key == GLFW_KEY_F && (action == GLFW_REPEAT || action == GLFW_PRESS))
		leftright-=0.1f;

	if (key == GLFW_KEY_G && (action == GLFW_REPEAT || action == GLFW_PRESS))
		frontback+=0.1f;

	if (key == GLFW_KEY_B && (action == GLFW_REPEAT || action == GLFW_PRESS))
		frontback-=0.1f;

	if (key == 334 && (action == GLFW_PRESS || action == GLFW_REPEAT))
		createSphere(ballRand(1.0f-radius), ballRand(velocity), elapsedTime);

	if(glfwGetKey( window, GLFW_KEY_T ) == GLFW_PRESS)
		texture = (texture+1)%2;

	if(glfwGetKey( window, GLFW_KEY_A ) == GLFW_PRESS)
		shading = (shading+1)%2;

	if(glfwGetKey( window, GLFW_KEY_M ) == GLFW_PRESS)
		showtraj = (showtraj+1)%2;

	if(glfwGetKey( window, GLFW_KEY_P ) == GLFW_PRESS)
		dex = (dex+1)%10;

	if(key == GLFW_KEY_MINUS && (action == GLFW_PRESS || action == GLFW_REPEAT))
		destroySphere();

	if(key == GLFW_KEY_L && action == GLFW_PRESS )
		light = (light+1)%4;
}  
//Handle window resizing
void window_size_callback(GLFWwindow* window, int width, int height)
{
	if (height==0)										// Prevent A Divide By Zero By
	{
		height=1;										// Making Height Equal One
	}

	glViewport(0,0,width,height);						// Reset The Current Viewport

	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix

	// Calculate The Aspect Ratio Of The Window
	gluPerspective(65.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();									// Reset The Modelview Matrix
}

int main( void )
{
	////////////////////////////////  INITIALIZATION  /////////////////////////////////////
	// Initialise GLFW
	if( !glfwInit() )
	{
		exit(EXIT_FAILURE);
	}

	glfwWindowHint(GLFW_SAMPLES, 4);

	GLFWwindow* window;

	window = glfwCreateWindow( width, height, "SAVVAS", NULL, NULL);

	//If the window couldn't be created  
    if (!window)  
    {  
        fprintf( stderr, "Failed to open GLFW window.\n" );  
        glfwTerminate();  
        exit(EXIT_FAILURE);  
    }  

	//This function makes the context of the specified window current on the calling thread.   
    glfwMakeContextCurrent(window);  

    //Initialize GLEW  
    GLenum err = glewInit();  
	//If GLEW hasn't initialized  
    if (err != GLEW_OK)   
    {  
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));  
        return -1;  
    }  

	/* Init opengl state */
	if(!InitGL())
		glfwTerminate();

    //Sets the key callback  
    glfwSetKeyCallback(window, key_callback);  
	//Sets the resize 
 	glfwSetWindowSizeCallback(window, window_size_callback);
	//Set the error callback  
    glfwSetErrorCallback(error_callback);  

	//////////////////////////////////  PROGRAM BEGIN  ///////////////////////////////////////
	std::srand(time(0));

	sphere.reserve(NUMSPHERES);

	ballTexture = loadBMP_custom("ball.bmp");
	//wallTexture = loadBMP_custom("walls.bmp");
	//floorTexture = loadBMP_custom("floor.bmp");

	velocity = 2.0f;
	
	elapsedTime = (float)glfwGetTime();
	
	// CREATION OF 4 BIG BALLS
	/*
	radius = 0.3f;
	createSphere(
		vec3(1.0f-radius, 1.0f-radius, -1.0f+radius),
		vec3(-0.5f, -0.5f, 0.0f),
		elapsedTime,
		vec3(0.8f,0.8f,0.8f)
	);
	createSphere(
		vec3(1.0f-4*radius, 1.0f-radius, -1.0f+radius),
		vec3(-0.5f, 0.0f, 0.0f),
		elapsedTime
	);
	
	createSphere(
		vec3(1.0f-4*radius, -1.0f+radius, -1.0f+radius),
		vec3(-0.5f, 0.0f, 0.0f),
		elapsedTime,
		vec3(1.0f,1.0f,1.0f)
	);
	createSphere(
		vec3(1.0f-radius, -1.0f+radius, -1.0f+radius),
		vec3(-0.5f, 0.0f, 0.0f),
		elapsedTime
	);
	*/
	// CREATION OF RANDOM P^3 BALLS WITH SAME SPEED
	
	radius = 0.05f;
	int p = 4;
	for (int k = 0; k<p; k++){
		for (int j = 0; j<p; j++){
			for (int l = 0; l<p; l++){
				createSphere(
					vec3(1.0f-radius-4*radius*k, 1.0f-radius-4*radius*j, 1.0f-radius-4*radius*l),
					vec3(-1.0f, 0.0f, 0.0f),
					//ballRand(velocity),
					elapsedTime,
					vec3(0.4f, 0.4f, 0.6f)
				);
			}
		}
	}

	//currentTime = glfwGetTime();

	do{
		//newTime = glfwGetTime();
		//frameTime = newTime - currentTime;

		//if ( frameTime > 0.25 )
		//	frameTime = 0.25f;

		//currentTime = newTime;

		//accumulator += frameTime;

		//while ( accumulator >= dt ){

			//Get and organize events, like keyboard and mouse input, window resizing, etc...  
			glfwPollEvents();

			//Update size of window
			glfwGetWindowSize(window, &width, &height);

			//Collision checking
			checkCollision();

		//	elapsedTime += dt;
		//	accumulator -= dt;
		//}

		//Draw objects
		DrawGLScene();

		// Swap buffers
		glfwSwapBuffers(window);

		elapsedTime = glfwGetTime();		
	}
	while(!glfwWindowShouldClose(window));

	//Close OpenGL window and terminate GLFW  
    glfwDestroyWindow(window);  
    //Finalize and clean up GLFW  
    glfwTerminate();  
  
    exit(EXIT_SUCCESS);  
}