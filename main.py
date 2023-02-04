import os
import mysql.connector
import matplotlib.pyplot as plt
import xml.etree.ElementTree as XmlParser


def load_data_to_db(db, folder_path):
    cursor = db.cursor()  # create a cursor to execute queries

    cursor.execute("DROP DATABASE IF EXISTS meca_project_tracking;")  # drop the database if it exists
    cursor.execute("CREATE DATABASE meca_project_tracking;")  # create the database
    cursor.execute("USE meca_project_tracking;")  # select the database

    counter = 1
    for folders, sub_folders, files in os.walk(folder_path):    # walk through all the files in the folder
        for file in files:  # for each file
            if ".Trk" in file.title():  # if the file is a tracking file
                cursor.execute(
                    "CREATE TABLE tracking_" +
                    str(counter) +
                    "(time double NOT NULL, px double NOT NULL, py double NOT NULL, "
                    "vx double  NOT NULL, vy double NOT NULL);"
                )  # create a table for the tracking file

                xml_tree_root = XmlParser.parse(os.path.join(folders, file)).getroot()  # load file in parser

                t, px, py, vx, vy = [0], [], [], [0], [0]  # create lists for time, x and y positions

                for x_pos in xml_tree_root.findall(".//property[@name='x']"):  # get all the x positions
                    px.append(float(x_pos.text))

                for y_pos in xml_tree_root.findall(".//property[@name='y']"):  # get all the y positions
                    py.append(float(y_pos.text))

                for i in range(len(px)):  # create the time list
                    t.append(t[i] + float(xml_tree_root.find(".//property[@name='delta_t']").text) / 1000)

                for i in range(1, len(px) - 1):  # create the x velocity list
                    vx.append((px[i] - px[i - 1]) / (t[i] - t[i - 1]))

                for i in range(1, len(py) - 1):  # create the y velocity list
                    vy.append((py[i] - py[i - 1]) / (t[i] - t[i - 1]))

                x_max = max(px)  # get the max x position
                for i in range(len(px)):  # for each x position
                    px[i] = x_max - px[i]  # invert the x position

                y_max = max(py)  # get the max y position
                for i in range(len(py)):  # for each y position
                    py[i] = y_max - py[i]  # invert the y position

                cursor.execute(("INSERT INTO tracking_" + str(counter) + " (time, px, py, vx, vy) VALUES " +
                                str([(ti, pxi, pyi, vxi, vyi) for ti, pxi, pyi, vxi, vyi in zip(t, px, py, vx, vy)])
                                + ";").replace("[", "").replace("]", "")
                               )  # insert the data into the table

                counter += 1


def plot_tracking(db, index):
    cursor = db.cursor()  # create a cursor to execute queries

    cursor.execute("USE meca_project_tracking;")  # select the database
    cursor.execute("SELECT * FROM tracking_" + str(index) + ";")  # select the data from the table

    t, px, py, vx, vy = zip(*cursor.fetchall())  # unpack the data

    # Plot the path of the object

    plt.plot(px, py)
    plt.title("Path of the object (mm)")
    plt.show()

    # Plot the x and y positions as a function of time

    px_plot = plt.subplot2grid((2, 1), (0, 0))
    py_plot = plt.subplot2grid((2, 1), (1, 0))

    px_plot.plot(t, px, label="X position (mm)")
    px_plot.set_title("Tracking " + str(index) + " position in terms of time")
    px_plot.legend()

    py_plot.plot(t, py, label="Y position (mm)")
    py_plot.legend()

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

    # Plot the x and y velocities as a function of time

    vx_plot = plt.subplot2grid((2, 1), (0, 0))
    vy_plot = plt.subplot2grid((2, 1), (1, 0))

    vx_plot.plot(t, vx, label="X velocity (mm/s)")
    vx_plot.set_title("Tracking " + str(index) + " velocity in terms of time")
    vx_plot.legend()

    vy_plot.plot(t, vy, label="Y velocity (mm/s)")
    vy_plot.legend()

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


if __name__ == "__main__":
    database = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database=""
    )

    load_data_to_db(database, os.getcwd() + "/tracking")
    plot_tracking(database, 1)

    database.close()
