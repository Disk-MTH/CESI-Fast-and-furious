import os
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import xml.etree.ElementTree as XmlParser


def load_data_to_db(db_cursor, folder_path):
    db_cursor.fetchall()  # clear the cursor
    counter = 1
    for folders, sub_folders, files in os.walk(folder_path):  # walk through all the files in the folder
        for file in files:  # for each file
            if ".Trk" in file.title():  # if the file is a tracking file
                db_cursor.execute(
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

                db_cursor.execute(("INSERT INTO tracking_" + str(counter) + " (time, px, py, vx, vy) VALUES " +
                                   str([(ti, pxi, pyi, vxi, vyi) for ti, pxi, pyi, vxi, vyi in zip(t, px, py, vx, vy)])
                                   + ";").replace("[", "").replace("]", "")
                                  )  # insert the data into the table

                counter += 1


def load_data_from_db(db_cursor, tracking_index):
    db_cursor.fetchall()  # clear the cursor
    db_cursor.execute("SELECT * FROM tracking_" + str(tracking_index) + ";")  # select the data from the table
    return zip(*db_cursor.fetchall())  # unpack the data


def plot_tracking(db_cursor, index):
    t, px, py, vx, vy = load_data_from_db(db_cursor, index)  # load the data from the database

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


def get_outstanding_values_per_tracking(db_cursor, index):
    t, px, py, vx, vy = load_data_from_db(db_cursor, index)  # load the data from the database

    vy_at_looping_top = min(vy)  # get the minimum y velocity
    vx_at_looping_top = vx[vy.index(vy_at_looping_top)]  # get the x velocity at the minimum y velocity
    v_at_looping_top = np.sqrt(vx_at_looping_top ** 2 + vy_at_looping_top ** 2)

    t_at_looping_top = t[vy.index(vy_at_looping_top)]  # get the time at the minimum y velocity

    vy_at_slope_end = max(vy[:t.index(t_at_looping_top)])  # get the maximum y velocity
    vx_at_slope_end = vx[vy.index(vy_at_slope_end)]  # get the x velocity at the maximum y velocity
    v_at_slope_end = np.sqrt(vx_at_slope_end ** 2 + vy_at_slope_end ** 2)

    vy_at_looping_end = max(vy[t.index(t_at_looping_top):])  # get the maximum y velocity
    vx_at_looping_end = vx[vy.index(vy_at_looping_end)]  # get the x velocity at the maximum y velocity
    v_at_looping_end = np.sqrt(vx_at_looping_end ** 2 + vy_at_looping_end ** 2)

    return v_at_slope_end, v_at_looping_top, v_at_looping_end


def get_outstanding_values(db_cursor):
    db_cursor.fetchall()  # clear the cursor
    db_cursor.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
                   f"WHERE TABLE_SCHEMA = '{schema_name}';")  # query the number of tables
    tracking_count = db_cursor.fetchall()[0][0]  # get the number of tables

    # get the outstanding values for each tracking
    slope_end_velocities, \
        looping_top_velocities, \
        looping_end_velocities = zip(*[get_outstanding_values_per_tracking(db_cursor, i) for i in range(1, tracking_count + 1)])

    # calculate the average and uncertainty of the outstanding values
    slope_end_velocity_average = np.average(slope_end_velocities)
    slope_end_velocity_uncertainty = np.std(slope_end_velocities) / np.sqrt(len(slope_end_velocities))

    looping_top_velocity_average = np.average(looping_top_velocities)
    looping_top_velocity_uncertainty = np.std(looping_top_velocities) / np.sqrt(len(looping_top_velocities))

    looping_end_velocity_average = np.average(looping_end_velocities)
    looping_end_velocity_uncertainty = np.std(looping_end_velocities) / np.sqrt(len(looping_end_velocities))

    return (slope_end_velocity_average, slope_end_velocity_uncertainty), \
        (looping_top_velocity_average, looping_top_velocity_uncertainty), \
        (looping_end_velocity_average, looping_end_velocity_uncertainty)


if __name__ == "__main__":
    schema_name = "meca_project_tracking"

    database = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root"

    )

    cursor = database.cursor()
    cursor.execute("DROP DATABASE IF EXISTS meca_project_tracking;")  # drop the database if it exists
    cursor.execute("CREATE DATABASE meca_project_tracking;")  # create the database
    cursor.execute(f"USE {schema_name};")  # select the database

    load_data_to_db(cursor, os.getcwd() + "/tracking")  # load the data to the database



    database.close()
