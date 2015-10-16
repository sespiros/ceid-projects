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

#define NUMSPHERES 1000

using namespace glm;

class Sphere {
public:
	GLUquadric *body;
	float t0;
	vec3 q0;
	vec3 speedVec;
	vec3 color;
	int index, cols;

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
int showtraj=1, id=0;
std::vector<Sphere> sphere;
std::vector<Sphere>::iterator it;
std::stack<int> freepos;
vec3 position;
std::priority_queue<CollisionEvent, std::vector<CollisionEvent>, CompareCollisionTimes> events;

//Utility functions
int toggleFullscreen(GLFWwindow* window, int a)
{
    //glfwDestroyWindow(window); 

	//if( !glfwCreateWindow( width, height, "Test window", (a==0)?NULL:glfwGetPrimaryMonitor(), NULL))
	//{
	//	exit(EXIT_FAILURE);
	//}

	return 1;
}

void calculateWallCollisionFor(std::vector<Sphere>::iterator iter)
{
	float tx=1000.0f, ty=1000.0f, tz=1000.0f;
	float min = 1000.0f;

	iter->newSpeedVec = iter->speedVec;

	vec3 p = iter->q0;
		
	//Check collision with walls
	if ( iter->speedVec.x > 0.0f )	//possible Collision with right wall(left as we see it)
	{
		tx = (1.0f-radius-p.x)/iter->speedVec.x;
		//printf("possible left wall collision at %f secs\n", tx);
		min = (tx<min)?tx:min;
	}
	else if ( iter->speedVec.x < 0.0f )
	{
		tx = -(1.0f-radius+p.x)/iter->speedVec.x;
		//printf("possible right wall collision at %f secs\n", tx);
		min = (tx<min)?tx:min;
	}//-----------------------------------------------------------------------
	if ( iter->speedVec.y > 0.0f )
	{
		ty = (1.0f-radius-p.y)/iter->speedVec.y;
		//printf("possible top wall collision at %f secs\n", ty);
		min = (ty<min)?ty:min;
	}
	else if ( iter->speedVec.y < 0.0f )
	{
		ty = -(1.0f-radius+p.y)/iter->speedVec.y;
		//printf("possible bottom wall collision at %f secs\n", ty);
		min = (ty<min)?ty:min;
	}//-----------------------------------------------------------------------
	if ( iter->speedVec.z > 0.0f )
	{
		tz = (1.0f-radius-p.z)/iter->speedVec.z;
		//printf("possible back wall collision at %f secs\n", tz);
		min = (tz<min)?tz:min;
	}
	else if ( iter->speedVec.z < 0.0f )
	{
		tz = -(1.0f-radius+p.z)/iter->speedVec.z;
		//printf("possible front wall collision at %f secs\n", tz);
		min = (tz<min)?tz:min;
	}//-----------------------------------------------------------------------

	//=====================================================================================
	if (tx==min){
		//printf("==Ball %d will hit left or right wall first\n",i);
		iter->newSpeedVec.x *=-1;
	}
	if (ty==min){
		//printf("==Ball %d will hit bottom or top wall first\n",i);
		iter->newSpeedVec.y *=-1;
	}
	if (tz==min){
		//printf("==Ball %d will hit front or back wall first\n",i);
		iter->newSpeedVec.z *=-1;
	}

	iter->nextCollision = min + elapsedTime;
	iter->lastCollision = elapsedTime;

	//printf("--------------------------------min==%f\n\n", min);
}

std::vector<Sphere>::iterator
	calculateBallCollisionFor(std::vector<Sphere>::iterator iter){
	int i, j;
	vec3 q, v, factor, p1, p2;
	float a, b, c, D, t1, t2, t, dist, min;
	std::vector<Sphere>::iterator iterj = sphere.end();

	i = iter->index;

	min = sphere[i].nextCollision;
	it = sphere.begin();
	while(it!=sphere.end()){
		j = it->index;
		if(j!=i){
			p1 = sphere[i].q0 + (float)(elapsedTime - sphere[i].t0)*sphere[i].speedVec;
			p2 = sphere[j].q0 + 
				(float)(elapsedTime - sphere[j].t0)*sphere[j].speedVec;
			
			q = p1 - p2;
			v = sphere[i].speedVec - sphere[j].speedVec;

			a = dot(v, v);
			b = 2*dot(q, v);
			c = dot(q, q) - 4*radius*radius;

			D = b*b - 4*a*c;

			if ( D >= 0 && b < 0 ) {
				t1 = (-b + sqrt(D))/(2*a);
				t2 = (-b - sqrt(D))/(2*a);
			
				t = (t1<t2)?t1:t2;

				if( t < min && t>0 ){
					min = t;
					iterj = it;	
				}
			}
		}
		it++;
	}
	if (iterj != sphere.end()){
		p1 = iter->q0	+ (min+elapsedTime - iter->t0)	*	iter->speedVec;
		p2 = iterj->q0	+ (min+elapsedTime - iterj->t0)	*	iterj->speedVec;
		q = p1 - p2;
		v = iter->speedVec - iterj->speedVec;

		factor = q*dot(v,q)/(4*radius*radius);

		iter->newSpeedVec = iter->speedVec - factor;
		iter->nextCollision = min + elapsedTime;
		iter->lastCollision = elapsedTime;

		iterj->newSpeedVec = iterj->speedVec + factor;
		iterj->nextCollision = min + elapsedTime;
		iterj->lastCollision = elapsedTime;
	}
	
	return iterj;
}

void calculateCollision(std::vector<Sphere>::iterator iter){
	CollisionEvent col1;//, col2;
	std::vector<Sphere>::iterator iterj = sphere.end();
	calculateWallCollisionFor(iter);
	iterj = calculateBallCollisionFor(iter);

	//sphere[i].cols++;
	if (iterj != sphere.end()){
		//sphere[balltoball].cols++;
		col1 = CollisionEvent(iter->index, iterj->index, iter->nextCollision, elapsedTime,  iter->cols, iterj->cols);
	}else
		col1 = CollisionEvent(iter->index, -1, iter->nextCollision, elapsedTime, iter->cols, 0);
	events.push(col1);
}

int checkOutdated(CollisionEvent event){
	int i, j;

	i = event.i;
	j = event.j;

	if( j==-1 ){ //only one ball
		if (sphere[i].cols == event.icols)
			return 0;
	}else{		//two balls
		if (sphere[i].cols == event.icols && sphere[j].cols == event.jcols)
			return 0;
	}

	return 1;
}

void checkCollision(){
	int i, j;
	while( elapsedTime >= events.top().timeOccuring)
	{
		i = events.top().i;

		if( checkOutdated( events.top() ) ){
			//printf("POOPZIES of %d, timeInserted=%f, lastCollition=%f\n",
			//	i, events.top().timeInserted, sphere[i].lastCollision );
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
			//printf("BALLZIES of %d, timeInserted=%f, lastCollition=%f\n",
			//	i, events.top().timeInserted, sphere[i].lastCollision );
		}else{
			sphere[i].q0 += (events.top().timeOccuring - sphere[i].t0) * sphere[i].speedVec;
			sphere[i].speedVec = sphere[i].newSpeedVec;
			sphere[i].t0 = events.top().timeOccuring;

			sphere[i].cols++;

			events.pop();
			calculateCollision(i);
			//printf("WALLZIES of %d, timeInserted=%f, lastCollition=%f\n",
			//	i, events.top().timeInserted, sphere[i].lastCollision );
		}
	}
	it = sphere.begin();
	while(it!=sphere.end()){
		position = it->q0 + (float)(elapsedTime - it->t0) * it->speedVec;
		if(abs(position.x)>0.96f || abs(position.y)>0.96f || abs(position.z)>0.96f){
			printf("WHAT THE FUCK MOAR BUGS PLS!!!!!AAAAAAAAAAARRGGH\n");
		}
		it++;
	}
}

void createSphere(vec3 startPos, vec3 velocity, float t0){
	if(!sphere.empty()){
		sphere.push_back(Sphere(startPos, velocity, t0, id));
		calculateCollision(sphere.end());
		id++;
	}
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
	it = sphere.begin();
	while(it != sphere.end()){

		//glColor3f(0.7f,0.7f,0.7f); 
		glColor3f(0.2f*(2*it->index+1),0.2f*(2*it->index+1),0.2f*(2*it->index+1));

		position = it->q0 + (float)(elapsedTime -it->t0) * it->speedVec;
		glPushMatrix();
		glTranslatef( position.x, position.y ,position.z );
		gluSphere(it->body, radius, 30, 30);

		if(abs(position.x)>1.0f || abs(position.y)>1.0f || abs(position.z)>1.0f){
			glColor3f(1.0f, 0.0f, 0.0f); //DAT ROGUE BALL
			it = sphere.erase(it);
		}else{
			it++;
		}


		glPopMatrix();
	}

	//draw Lines
	it = sphere.begin();
	if(showtraj){
	while(it != sphere.end()){
		glBegin(GL_LINES);
			glColor3f(0.5f,0.5f,0.5f);
			glVertex3f(it->q0.x, it->q0.y, it->q0.z);
			glVertex3f(it->q0.x + 2*it->speedVec.x,
					   it->q0.y + 2*it->speedVec.y,
					   it->q0.z + 2*it->speedVec.z);
		glEnd();
		it++;
	}
	}
	return true;
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

	if (key == 334 && (action == GLFW_REPEAT || action == GLFW_PRESS))
		createSphere(
		//ballRand(1.0f-radius),
		vec3((float)(rand()%95)/100, (float)(rand()%95)/100, -1.0f+radius),
		//ballRand(speed), 
		vec3((float)(rand()%100)/100, (float)(rand()%100)/100, 0.0f),
		elapsedTime);

	if(glfwGetKey( window, GLFW_KEY_P ) == GLFW_PRESS)
		showtraj = (showtraj+1)%2;

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


	radius = 0.10;

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
	createSphere(
		vec3(1.0f-7*radius, 1.0f-radius, -1.0f+radius),
		vec3(-0.5f, 0.0f, 0.0f),
		elapsedTime
	);
	createSphere(
		vec3(1.0f-10*radius, 1.0f-radius, -1.0f+radius),
		vec3(-0.5f, 0.0f, 0.0f),
		elapsedTime
	);


	/*
	int p = 5;
	for (int k = 0; k<p; k++){
		for (int j = 0; j<p; j++){
			for (int l = 0; l<p; l++){
				createSphere(
					vec3(1.0f-radius-4*radius*k, 1.0f-radius-4*radius*j, 1.0f-radius-4*radius*l),
					vec3(-1.0f, 0.0f, 0.0f),
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

		elapsedTime = glfwGetTime();
	}
	while(!glfwWindowShouldClose(window));

	//Close OpenGL window and terminate GLFW  
    glfwDestroyWindow(window);  
    //Finalize and clean up GLFW  
    glfwTerminate();  
  
    exit(EXIT_SUCCESS);  
}
