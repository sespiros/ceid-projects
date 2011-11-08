//TEST COMMIT VIA QT
////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include <SFML/Graphics.hpp>


////////////////////////////////////////////////////////////
/// Entry point of application
///
/// \return Application exit code
///
////////////////////////////////////////////////////////////
int main()
{
    // Create the main window
    sf::RenderWindow App(sf::VideoMode(800, 600), "SFML Views");

    // Create a sprite for the background
    sf::Image BackgroundImage;
    if (!BackgroundImage.LoadFromFile("c:/background.jpg"))
        return EXIT_FAILURE;
    sf::Sprite Background(BackgroundImage);

    // Create another sprite for the cursor
   sf::Image CursorImage;
   if (!CursorImage.LoadFromFile("c:/cursor.tga"))
        return EXIT_FAILURE;
    sf::Sprite Cursor(CursorImage);

    // Resize the background, so that it's much bigger than the window
    Background.Resize(2000, 2000);

    // Define a text for displaying some instructions
    sf::String Text("Use the arrow keys to move the camera\nUse the + and - keys to zoom and unzoom");
    Text.Move(10, 500);
    Text.SetColor(sf::Color::White);

    // Create a view with the same size as the window, located somewhere in the center of the background
    sf::Vector2f Center(1000, 1000);
    sf::Vector2f HalfSize(400, 300);
    sf::View View(Center, HalfSize);

    // To start, put the cursor at the center
   Cursor.SetCenter(32, 32);
    Cursor.SetPosition(Center);

    // Start game loop
    while (App.IsOpened())
    {
        // Process events
        sf::Event Event;
        while (App.GetEvent(Event))
        {
            // Close window : exit
            if (Event.Type == sf::Event::Closed)
                App.Close();

            // Escape key : exit
            if ((Event.Type == sf::Event::KeyPressed) && (Event.Key.Code == sf::Key::Escape))
                App.Close();
        }

        // Move the view using arrow keys
        float Offset = 200.f * App.GetFrameTime();
        if (App.GetInput().IsKeyDown(sf::Key::Up))    View.Move( 0,      -Offset);
        if (App.GetInput().IsKeyDown(sf::Key::Down))  View.Move( 0,       Offset);
        if (App.GetInput().IsKeyDown(sf::Key::Left))  View.Move(-Offset,  0);
        if (App.GetInput().IsKeyDown(sf::Key::Right)) View.Move( Offset,  0);

        // Zoom and unzoom using + and - keys
        if (App.GetInput().IsKeyDown(sf::Key::Add))      View.Zoom(1.001f);
        if (App.GetInput().IsKeyDown(sf::Key::Subtract)) View.Zoom(0.999f);

        // Set the view
        App.SetView(View);

        // Clear screen
        App.Clear();

        // Draw background
        App.Draw(Background);

        // Draw cursor
        sf::Vector2f CursorPos = App.ConvertCoords(App.GetInput().GetMouseX(), App.GetInput().GetMouseY());
        Cursor.SetPosition(CursorPos);
        App.Draw(Cursor);

        // Reset to the default view to draw the interface
        App.SetView(App.GetDefaultView());

        // Draw some instructions
        App.Draw(Text);

        // Finally, display rendered frame on screen
        App.Display();
    }

    return EXIT_SUCCESS;
}
