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

#define NUMSPHERES 4

using namespace glm;

class Sphere {
public:
	GLUquadric *body;
	float t0;
	vec3 q0;
	vec3 q1;
	vec3 speedVec;
	vec3 color;
	int index, cols;
	bool empty;

	float nextCollision, lastCollision;
	vec3 newSpeedVec;

	Sphere(){}
	Sphere(vec3 q, vec3 speedV, float t, int ind){
		q0 = q;
		speedVec = speedV;
		t0 = t;
		index = ind;
		body = gluNewQuadric();
		cols = 0;
		empty = false;
	}
	void setAttributes(vec3 q, vec3 speedV, float t, int ind){
		q0 = q;
		speedVec = speedV;
		t0 = t;
		index = ind;
		body = gluNewQuadric();
		cols = 0;
		empty = false;
	}

};

class CollisionEvent {
public:
	int i,j;
	float timeOccuring, timeInserted;
	int icols, jcols;

	CollisionEvent(){}
	CollisionEvent (int sphereI, int sphereJ, float nextCollision, float inserted, int cols, int cols2){
		i = sphereI;
		j = sphereJ;
		timeOccuring = nextCollision;
		timeInserted = inserted;
		icols = cols;
		jcols = cols2;
	}
};

class CompareCollisionTimes {
public:
	bool operator() (CollisionEvent s1, CollisionEvent s2)
	{
		//if (s1.i==3 && s1.j==2 && s1.timeOccuring>5.30f)
		//	printf("tsiriosliiiiiiii?\n");
		if (s1.timeOccuring > s2.timeOccuring)
			return true;
		return false;
	}
};

GLsizei		width  = 1024;
GLsizei		height = 768;
int fullscreen = 0;
float speed, radius = 0.4f, frontback=0.0f, updown=0.0f, leftright=0.0f;
float collisionTime;
float elapsedTime;
float minmin;
int numSpheres = 0, showtraj=1;
std::vector<Sphere>::iterator sphereIterInit;
std::vector<Sphere> sphere;
std::stack<int> freepos;
vec3 position;
std::priority_queue<CollisionEvent, std::vector<CollisionEvent>, CompareCollisionTimes> events;

//COLLISION FUNCTIONS
void calculateWallCollisionFor(int i)
{
	float tx=1000.0f, ty=1000.0f, tz=1000.0f;
	float min = 1000.0f;

	sphere[i].newSpeedVec = sphere[i].speedVec;

	vec3 p = sphere[i].q0;
		
	//Check collision with walls
	if ( sphere[i].speedVec.x > 0.0f )	//possible Collision with right wall(left as we see it)
	{
		tx = (1.0f-radius-p.x)/sphere[i].speedVec.x;
		//printf("possible left wall collision at %f secs\n", tx);
		min = (tx<min)?tx:min;
	}
	else if ( sphere[i].speedVec.x < 0.0f )
	{
		tx = -(1.0f-radius+p.x)/sphere[i].speedVec.x;
		//printf("possible right wall collision at %f secs\n", tx);
		min = (tx<min)?tx:min;
	}//-----------------------------------------------------------------------
	if ( sphere[i].speedVec.y > 0.0f )
	{
		ty = (1.0f-radius-p.y)/sphere[i].speedVec.y;
		//printf("possible top wall collision at %f secs\n", ty);
		min = (ty<min)?ty:min;
	}
	else if ( sphere[i].speedVec.y < 0.0f )
	{
		ty = -(1.0f-radius+p.y)/sphere[i].speedVec.y;
		//printf("possible bottom wall collision at %f secs\n", ty);
		min = (ty<min)?ty:min;
	}//-----------------------------------------------------------------------
	if ( sphere[i].speedVec.z > 0.0f )
	{
		tz = (1.0f-radius-p.z)/sphere[i].speedVec.z;
		//printf("possible back wall collision at %f secs\n", tz);
		min = (tz<min)?tz:min;
	}
	else if ( sphere[i].speedVec.z < 0.0f )
	{
		tz = -(1.0f-radius+p.z)/sphere[i].speedVec.z;
		//printf("possible front wall collision at %f secs\n", tz);
		min = (tz<min)?tz:min;
	}//-----------------------------------------------------------------------

	//=====================================================================================
	if (tx==min){
		//printf("==Ball %d will hit left or right wall first\n",i);
		sphere[i].newSpeedVec.x *=-1;
	}
	if (ty==min){
		//printf("==Ball %d will hit bottom or top wall first\n",i);
		sphere[i].newSpeedVec.y *=-1;
	}
	if (tz==min){
		//printf("==Ball %d will hit front or back wall first\n",i);
		sphere[i].newSpeedVec.z *=-1;
	}

	//check wall boundaries
	//p = sphere[i].q0 + (elapsedTime - sphere[i].t0)*sphere[i].speedVec;
	//if ( abs(p.x) > 1.0f-radius )
		
	//if ( abs(p.y) > 1.0f-radius )
		
	//if ( abs(p.z) > 1.0f-radius )

	sphere[i].nextCollision = min + elapsedTime;
	sphere[i].lastCollision = elapsedTime;

	//printf("--------------------------------min==%f\n\n", min);
}

int calculateBallCollisionFor(int i){
	int j=0, balltoball=-1;
	vec3 q, v, factor, p1, p2;
	float a, b, c, D, t1, t2, t, dist, min;

	float distX, distY, distZ;

	min = sphere[i].nextCollision;
	if (i==j)j++;
	while(j<numSpheres){
		if(j!=i){
			p1 = sphere[i].q0 + (elapsedTime - sphere[i].t0)*sphere[i].speedVec;
			p2 = sphere[j].q0 + (elapsedTime - sphere[j].t0)*sphere[j].speedVec;
			
			q = p1 - p2;
			v = sphere[i].speedVec - sphere[j].speedVec;

			a = dot(v, v);
			b = 2*dot(q, v);
			c = dot(q, q) - 4*radius*radius;

			D = b*b - 4*a*c;

			if ( D >= 0 ) {//b>0?
				t1 = (-b + sqrt(D))/(2*a);
				t2 = (-b - sqrt(D))/(2*a);
			
				//p1 = sphere[i].q0 + (t1 + elapsedTime - sphere[i].t0)*sphere[i].speedVec;
				//p2 = sphere[j].q0 + (t1 + elapsedTime - sphere[j].t0)*sphere[j].speedVec;
				
				t = (t2<t1)?t2:t1;
				

				if( t < min && t>0 ){
					printf("i=%d, j=%d, t1 = %f, t2 = %f\n", i, j, t1, t2);
					min = t;
					balltoball = j;	
				}
			}
		}
		j++;
	}
	j = balltoball;
	if (j!=-1){
		p1 = sphere[i].q0 + (min + elapsedTime - sphere[i].t0)*sphere[i].speedVec;
		p2 = sphere[j].q0 + (min + elapsedTime - sphere[j].t0)*sphere[j].speedVec;
		q = p1 - p2;
		v = sphere[i].speedVec - sphere[j].speedVec;

		factor = q*dot(v,q)/(4*radius*radius);

		sphere[i].newSpeedVec = sphere[i].speedVec - factor;
		sphere[i].nextCollision = min + elapsedTime;
		sphere[i].lastCollision = elapsedTime;

		sphere[j].newSpeedVec = sphere[j].speedVec + factor;
		sphere[j].nextCollision = min + elapsedTime;
		sphere[j].lastCollision = elapsedTime;
	}
	
	return balltoball;
}

void calculateCollision(int i){
	CollisionEvent col1;//, col2;
	int balltoball = -1;
	calculateWallCollisionFor(i);
	balltoball = calculateBallCollisionFor(i);

	//sphere[i].cols++;
	if (balltoball!=-1){
		//sphere[balltoball].cols++;
		col1 = CollisionEvent(i, balltoball, sphere[i].nextCollision, elapsedTime,  sphere[i].cols, sphere[balltoball].cols);
	}else
		col1 = CollisionEvent(i, -1, sphere[i].nextCollision, elapsedTime, sphere[i].cols, 0);
	events.push(col1);
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
/*
int checkOutdated(CollisionEvent event){
	int i, j;

	i = event.i;
	j = event.j;

	if (sphere[i].empty)
		return 0;

	if( j==-1 ){ //only one ball
		if (sphere[i].cols != event.icols && event.timeOccuring<sphere[i].nextCollision)
			return 1;
	}else{		//two balls
		if (sphere[j].empty)
			return 0;
		if (sphere[i].cols == event.icols && sphere[j].cols == event.jcols &&
			event.timeOccuring<sphere[i].nextCollision && event.timeOccuring<sphere[j].nextCollision)
			return 1;
	}

	return 0;
}*/

void checkCollision(){
	int i, j;
	if ( events.size()!=0){
	while( elapsedTime >= events.top().timeOccuring)
	{
		i = events.top().i;

		if( checkOutdated( events.top() ) ){
			printf("---POOPZIES of %d, timeInserted=%f, lastCollition=%f\n",
				i, events.top().timeInserted, sphere[i].lastCollision );
			events.pop();
		} else if (events.top().j!=-1){
			j = events.top().j;

			sphere[i].q0 += (events.top().timeOccuring - sphere[i].t0) * sphere[i].speedVec;
			sphere[i].speedVec = sphere[i].newSpeedVec;
			sphere[i].t0 = events.top().timeOccuring;

			sphere[j].q0 += (events.top().timeOccuring - sphere[j].t0) * sphere[j].speedVec;
			sphere[j].speedVec = sphere[j].newSpeedVec;
			sphere[j].t0 = events.top().timeOccuring;

			sphere[i].cols++;
			sphere[j].cols++;

			events.pop();
			calculateCollision(i);
			calculateCollision(j);
			printf("BALLZIES of %d, timeInserted=%f, lastCollition=%f\n",
				i, events.top().timeInserted, sphere[i].lastCollision );
		}else{
			sphere[i].q0 += (events.top().timeOccuring - sphere[i].t0) * sphere[i].speedVec;
			sphere[i].speedVec = sphere[i].newSpeedVec;
			sphere[i].t0 = events.top().timeOccuring;

			sphere[i].cols++;

			events.pop();
			calculateCollision(i);
			printf("WALLZIES of %d, timeInserted=%f, lastCollition=%f\n",
				i, events.top().timeInserted, sphere[i].lastCollision );
		}
	}
	//std::vector<Sphere>::iterator it = sphereIterInit;
	//for(it; it != sphere.end();it++){
	//	position = it->q0 + (float)(elapsedTime - it->t0) * it->speedVec;
	//	if(abs(position.x)>0.96f || abs(position.y)>0.96f || abs(position.z)>0.96f){
	//		printf("WHAT THE FUCK MOAR BUGS PLS!!!!!AAAAAAAAAAARRGGH\n");
	//	}
	//}


	minmin = (events.top().timeOccuring<minmin)?events.top().timeOccuring:minmin;
	}
}

void createSphere(vec3 startPos, vec3 velocity, float t0){
	int id;
	if(sphere.size()<NUMSPHERES){
		if(!freepos.empty()){
			id = freepos.top();
			freepos.pop();
			sphere[id].setAttributes(startPos, velocity, t0, id);
			sphereIterInit = sphereIterInit - 1;
		}else{
			id = numSpheres;
			numSpheres++;
			sphere.push_back(Sphere(startPos, velocity, t0, id));
			sphereIterInit = sphere.begin();
		}
		calculateCollision(id);
	}
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

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	return true;
}
//Drawing
int DrawGLScene(GLvoid)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	gluLookAt(0.0f,0.0f,-3.0f+frontback,0.0f+leftright,0.0f+updown,0.0f,0.0f,1.0f,0.0f);

	glBegin(GL_QUADS);                      // Draw A Quad
        glColor3f(0.0f,1.0f,0.0f);          // Set The Color To Green
		glVertex3f( 1.0f, 1.0f,-1.0f);          // Top Right Of The Quad (Top)
		glVertex3f(-1.0f, 1.0f,-1.0f);          // Top Left Of The Quad (Top)
		glVertex3f(-1.0f, 1.0f, 1.0f);          // Bottom Left Of The Quad (Top)
		glVertex3f( 1.0f, 1.0f, 1.0f);          // Bottom Right Of The Quad (Top)

		glColor3f(1.0f,1.0f,1.0f);          // Set The Color To Orange
		glVertex3f( 1.0f,-1.0f, 1.0f);          // Top Right Of The Quad (Bottom)
		glVertex3f(-1.0f,-1.0f, 1.0f);          // Top Left Of The Quad (Bottom)
		glVertex3f(-1.0f,-1.0f,-1.0f);          // Bottom Left Of The Quad (Bottom)
		glVertex3f( 1.0f,-1.0f,-1.0f);          // Bottom Right Of The Quad (Bottom)

		glColor3f(1.0f,0.0f,0.0f);          // Set The Color To Red
		glVertex3f( 1.0f, 1.0f, 1.0f);          // Top Right Of The Quad (Back)
		glVertex3f(-1.0f, 1.0f, 1.0f);          // Top Left Of The Quad (Back)
		glVertex3f(-1.0f,-1.0f, 1.0f);          // Bottom Left Of The Quad (Back)
		glVertex3f( 1.0f,-1.0f, 1.0f);          // Bottom Right Of The Quad (Back)

		glColor3f(0.0f,0.0f,1.0f);          // Set The Color To Blue
		glVertex3f(-1.0f, 1.0f, 1.0f);          // Top Right Of The Quad (Left)
		glVertex3f(-1.0f, 1.0f,-1.0f);          // Top Left Of The Quad (Left)
		glVertex3f(-1.0f,-1.0f,-1.0f);          // Bottom Left Of The Quad (Left)
		glVertex3f(-1.0f,-1.0f, 1.0f);          // Bottom Right Of The Quad (Left)

		glColor3f(1.0f,0.0f,1.0f);          // Set The Color To Violet
        glVertex3f( 1.0f, 1.0f,-1.0f);          // Top Right Of The Quad (Right)
        glVertex3f( 1.0f, 1.0f, 1.0f);          // Top Left Of The Quad (Right)
        glVertex3f( 1.0f,-1.0f, 1.0f);          // Bottom Left Of The Quad (Right)
        glVertex3f( 1.0f,-1.0f,-1.0f);          // Bottom Right Of The Quad (Right)
	glEnd();							    // Done Drawing The Quad 
	
	//drawSpheres
	std::vector<Sphere>::iterator it = sphereIterInit;
	for (it; it != sphere.end(); it++){

		//glColor3f(0.7f,0.7f,0.7f); 
		glColor3f(0.2f*(2*it->index+1),0.2f*(2*it->index+1),0.2f*(2*it->index+1));

		position = it->q0 + (elapsedTime - it->t0) * it->speedVec;
		glPushMatrix();
		glTranslatef( position.x, position.y ,position.z );
		if(abs(position.x)>1.1f-radius || abs(position.y)>1.0f-radius || abs(position.z)>1.0f-radius){
			glColor3f(1.0f, 0.0f, 0.0f); //DAT ROGUE BALL
		}
		gluSphere(it->body, radius, 30, 30);
		glPopMatrix();
	}

	//draw Lines
	it = sphereIterInit;
	if(showtraj){
	for (it; it != sphere.end(); it++){
		glBegin(GL_LINES);
			glColor3f(0.5f,0.5f,0.5f);
			glVertex3f(it->q0.x, it->q0.y, it->q0.z);
			glVertex3f(it->q0.x + 2*it->speedVec.x,
					   it->q0.y + 2*it->speedVec.y,
					   it->q0.z + 2*it->speedVec.z);
		glEnd();
	}
	}
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
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)  
		glfwSetWindowShouldClose(window, GL_TRUE);  

	// Toggle fullscreen
	if (glfwGetKey( window, GLFW_KEY_F11 ) == GLFW_PRESS )
		toggleFullscreen(window, ++fullscreen%2);

	if (key == GLFW_KEY_E && (action == GLFW_REPEAT || action == GLFW_PRESS))
		updown+=0.1;

	if (key == GLFW_KEY_D && (action == GLFW_REPEAT || action == GLFW_PRESS))
		updown-=0.1;

	if (key == GLFW_KEY_S && (action == GLFW_REPEAT || action == GLFW_PRESS))
		leftright+=0.1;

	if (key == GLFW_KEY_F && (action == GLFW_REPEAT || action == GLFW_PRESS))
		leftright-=0.1;

	if (key == GLFW_KEY_G && (action == GLFW_REPEAT || action == GLFW_PRESS))
		frontback+=0.1;

	if (key == GLFW_KEY_B && (action == GLFW_REPEAT || action == GLFW_PRESS))
		frontback-=0.1;

	if (key == 334 && (action == GLFW_PRESS ))//|| action == GLFW_REPEAT))
		createSphere(ballRand(1.0f-radius), ballRand(speed), elapsedTime);

	if(glfwGetKey( window, GLFW_KEY_P ) == GLFW_PRESS)
		showtraj = (showtraj+1)%2;

	if(key == GLFW_KEY_MINUS && (action == GLFW_PRESS))
		destroySphere();

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
	std::srand(time(0));

	//Set the error callback  
    glfwSetErrorCallback(error_callback);  

	// Initialise GLFW
	if( !glfwInit() )
	{
		exit(EXIT_FAILURE);
	}

	glfwWindowHint(GLFW_SAMPLES, 4);

	GLFWwindow* window;

	window = glfwCreateWindow( width, height, "Test window", NULL, NULL);

	//If the window couldn't be created  
    if (!window)  
    {  
        fprintf( stderr, "Failed to open GLFW window.\n" );  
        glfwTerminate();  
        exit(EXIT_FAILURE);  
    }  

	//This function makes the context of the specified window current on the calling thread.   
    glfwMakeContextCurrent(window);  
  
    //Sets the key callback  
    glfwSetKeyCallback(window, key_callback);  

	//Sets the resize 
 	glfwSetWindowSizeCallback(window, window_size_callback); 

    //Initialize GLEW  
    GLenum err = glewInit();  

	//If GLEW hasn't initialized  
    if (err != GLEW_OK)   
    {  
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));  
        return -1;  
    }  

	if(!InitGL())
		glfwTerminate();

	//speed = rand()%2 + 0.5;
	speed = 2.0f;

	elapsedTime = glfwGetTime();

	sphere.reserve(NUMSPHERES);

	radius = 0.4;
	
	createSphere(
		vec3(1.0f-radius, 1.0f-radius, -1.0f+radius),
		vec3(-0.5f, 0.0f, 0.0f),
		elapsedTime
	);
	createSphere(
		vec3(1.0f-4*radius, 1.0f-radius, -1.0f+radius),
		vec3(-0.5f, 0.0f, 0.0f),
		elapsedTime
	);
	/*createSphere(
		vec3(1.0f-7*radius, 1.0f-radius, -1.0f+radius),
		vec3(-0.5f, 0.0f, 0.0f),
		elapsedTime
	);
	createSphere(
		vec3(1.0f-10*radius, 1.0f-radius, -1.0f+radius),
		vec3(-0.5f, 0.0f, 0.0f),
		elapsedTime
	);*/


	/*
	int p = 5;
	for (int k = 0; k<p; k++){
		for (int j = 0; j<p; j++){
			for (int l = 0; l<p; l++){
				createSphere(
					vec3(1.0f-radius-4*radius*k, 1.0f-radius-4*radius*j, 1.0f-radius-4*radius*l),
					vec3(-1.0f, 0.0f, 0.0f),
					//ballRand(2.0f),
					elapsedTime
				);
			}
		}
	}*/


	/*
	createSphere(
		vec3(1.0f-radius, 0.0f, -1.0f+radius),
		vec3(-1.0f, 0.0f, 0.0f),
		elapsedTime
	);
	createSphere(
		vec3(-1.0f+radius, radius, -1.0f+radius),
		vec3(1.0f, 0.0f, 0.0f),
		elapsedTime
	);*/

	//Calculate collisions
	do{
		minmin = 1000.0f;
		//Get and organize events, like keyboard and mouse input, window resizing, etc...  
        glfwPollEvents();

		//Update size of window
		glfwGetWindowSize(window, &width, &height);

		//Collision checking
		checkCollision();

		//Draw objects
		DrawGLScene();

		// Swap buffers
		glfwSwapBuffers(window);

		//printf("Time interval: %f\n", glfwGetTime()-elapsedTime);

		elapsedTime = glfwGetTime();
		if (elapsedTime > minmin ) elapsedTime = minmin;

	}
	while(!glfwWindowShouldClose(window));

	//Close OpenGL window and terminate GLFW  
    glfwDestroyWindow(window);  
    //Finalize and clean up GLFW  
    glfwTerminate();  
  
    exit(EXIT_SUCCESS);  
}
