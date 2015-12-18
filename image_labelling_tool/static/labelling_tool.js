/*
The MIT License (MIT)

Copyright (c) 2015 University of East Anglia, Norwich, UK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Developed by Geoffrey French in collaboration with Dr. M. Fisher and
Dr. M. Mackiewicz.
 */
function LabellingTool() {
    /*
    Colour utility functions
     */
    var lighten_colour = function(rgb, amount) {
        var x = 1.0 - amount;
        return [Math.round(rgb[0]*x + 255*amount),
            Math.round(rgb[1]*x + 255*amount),
            Math.round(rgb[2]*x + 255*amount)];
    };

    var rgb_to_rgba_string = function(rgb, alpha) {
        return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + alpha + ')';
    };

    var compute_centroid_of_points = function(vertices) {
        var sum = [0.0, 0.0];
        var N = 0;
        for (var i = 0; i < vertices.length; i++) {
            var vtx = vertices[i];
            if (vtx !== null) {
                sum[0] += vtx.x;
                sum[1] += vtx.y;
                N += 1;
            }
        }
        if (N === 0) {
            return null;
        }
        else {
            var scale = 1.0 / N;
            return {
                x: sum[0] * scale,
                y: sum[1] * scale
            };
        }
    };



    /*
    Object ID table
     */
    var ObjectIDTable = function() {
        var self = {
            _id_counter: 1,
            _id_to_object: {}
        };

        self.get = function(id) {
            return self._id_to_object[id];
        };

        self.register = function(obj) {
            var id;
            if ('object_id' in obj  &&  obj.object_id !== null) {
                id = obj.object_id;
                self._id_counter = Math.max(self._id_counter, id+1);
                self._id_to_object[id] = obj;
            }
            else {
                id = self._id_counter;
                self._id_counter += 1;
                self._id_to_object[id] = obj;
            }
        };

        self.unregister = function(obj) {
            self._id_to_object[obj.object_id] = null;
            obj.object_id = null;
        };


        self.register_objects = function(object_array) {
            var obj, id, i;

            for (i = 0; i < object_array.length; i++) {
                obj = object_array[i];
                if ('object_id' in obj  &&  obj.object_id !== null) {
                    id = obj.object_id;
                    self._id_counter = Math.max(self._id_counter, id+1);
                    self._id_to_object[id] = obj;
                }
            }

            for (i = 0; i < object_array.length; i++) {
                obj = object_array[i];

                if ('object_id' in obj  &&  obj.object_id !== null) {

                }
                else {
                    id = self._id_counter;
                    self._id_counter += 1;
                    self._id_to_object[id] = obj;
                    obj.object_id = id;
                }
            }
        };

        return self;
    };


    /*
    Label header model

    This is the model that gets send back and forth between the frontend and the backend.
    It combines:
    - an array of labels
    - an image ID that identifies the image to which the labels belong
    - a complete flag that indicates if the image is done
     */

    var LabelHeaderModel = function(image_id, complete, labels) {
        var self = {image_id: image_id,
            complete: complete,
            labels: labels};

        return self;
    };

    var replace_label_header_labels = function(label_header, labels) {
        if (labels !== null) {
            if (labels.length !== undefined) {
                if (labels.length === 0) {
                    labels = [];
                }
            }
        }
        return LabelHeaderModel(label_header.image_id, label_header.complete, labels);
    };



    /*
    Abstract label model
     */
    var AbstractLabelModel = function() {
        var self = {
            label_type: null,
            label_class: null,
        };
        return self;
    };


    /*
    Create a polygonal label model

    vertices: list of pairs, each pair is [x, y]
     */
    var PolygonalLabelModel = function() {
        var self = AbstractLabelModel();
        self.label_type = 'polygon';
        self.vertices = [];
        return self;
    };


    /*
    Composite label model
     */
    var CompositeLabelModel = function() {
        var self = AbstractLabelModel();
        self.label_type = 'composite';
        self.components = [];

        return self;
    };




    var AbstractLabelEntityListener = function() {
        return {
            notify_entity_hover: function(entity, hover_state) {},
            notify_entity_select: function(entity, select_state) {},
            notify_entity_modified: function(entity) {}
        };
    };



    /*
    Abstract label entity
     */
    var AbstractLabelEntity = function(view, model) {
        var self = {
            model: model,
            _view: view,
            _hover: false,
            _selected: false,
            _entity_listeners: []
        };

        self.attach_entity_listener = function(listener) {
            self._entity_listeners.push(listener);
        };

        self.detach_mouse_in_handler = function(listener) {
            var index = self._entity_listeners.indexOf(listener);
            if (index !== -1) {
                self._entity_listeners.splice(index, 1);
            }
        };

        self.attach = function() {
        };

        self.detach = function() {
        };

        self.update = function() {
            for (var i = 0; i < self._entity_listeners.length; i++) {
                self._entity_listeners[i].notify_entity_modified(self);
            }
        };

        self.commit = function() {
        };

        self.hover = function(state) {
            self._hover = state;
            self.queue_redraw();

            for (var i = 0; i < self._entity_listeners.length; i++) {
                self._entity_listeners[i].notify_entity_hover(self, state);
            }
        };

        self.select = function(state) {
            self._selected = state;
            self.queue_redraw();

            for (var i = 0; i < self._entity_listeners.length; i++) {
                self._entity_listeners[i].notify_entity_select(self, state);
            }
        };

        self.notify_hide_labels_change = function(state) {
            self.queue_redraw();
        };

        self.get_label_class = function() {
            return self.model.label_class;
        };

        self.set_label_class = function(label_class) {
            self.model.label_class = label_class;
            view.commit_model(self.model);
            self.queue_redraw();
        };

        self.compute_centroid = function() {
            return null;
        };

        self.contains_point = function(point) {
            return false;
        };

        self.distance_to_point = function(point) {
            return null;
        };

        self.notify_model_destroyed = function(model_id) {
        };

        self.paint = function(ctx, scale_x, scale_y) {
        };

        self.queue_redraw = function() {
            self._view.queue_redraw();
        };

        return self;
    };


    /*
    Polygonal label entity
     */
    var PolygonalLabelEntity = function(view, polygonal_label_model) {
        var self = AbstractLabelEntity(view, polygonal_label_model);

        self._polyk_poly = [];
        self._centroid = null;

        self.attach = function() {
            self._update_poly();
        };

        self.detach = function() {
            self._polyk_poly = [];
        };

        self._update_poly = function() {
            self._polyk_poly = [];
            for (var i = 0; i < self.model.vertices.length; i++) {
                self._polyk_poly.push(self.model.vertices[i].x);
                self._polyk_poly.push(self.model.vertices[i].y);
            }
            self._centroid = compute_centroid_of_points(self.model.vertices);
        };

        var super_update = self.update;
        self.update = function() {
            super_update();
            self._update_poly();
            self.queue_redraw();
        };

        self.commit = function() {
            self._view.commit_model(self.model);
        };


        self.compute_centroid = function() {
            return self._centroid;
        };

        self.contains_point = function(point) {
            return PolyK.ContainsPoint(self._polyk_poly, point.x, point.y);
        };

        self.distance_to_point = function(point) {
            if (PolyK.ContainsPoint(self._polyk_poly, point.x, point.y)) {
                return 0.0;
            }
            else {
                var e = PolyK.ClosestEdge(self._polyk_poly, point.x, point.y);
                return e.dist;
            }
        };

        self.paint = function(ctx, scale_x, scale_y) {
            if (self.model.vertices.length > 0) {
                ctx.beginPath();
                ctx.moveTo(scale_x(self.model.vertices[0].x), scale_y(self.model.vertices[0].y));
                for (var i = 1; i < self.model.vertices.length; i++) {
                    ctx.lineTo(scale_x(self.model.vertices[i].x), scale_y(self.model.vertices[i].y));
                }
                ctx.closePath();

                if (!self._view.hide_labels) {
                    var fill_colour = self._view.colour_for_label_class(self.model.label_class);
                    if (self._hover) {
                        fill_colour = lighten_colour(fill_colour, 0.4);
                    }
                    ctx.fillStyle = rgb_to_rgba_string(fill_colour, 0.35);
                    ctx.fill();
                }

                var stroke_colour = self._selected ? [255,0,0] : [255,255,0];
                if (self._view.hide_labels) {
                    ctx.strokeStyle = rgb_to_rgba_string(stroke_colour, 0.2);
                }
                else {
                    ctx.strokeStyle = rgb_to_rgba_string(stroke_colour, 0.5);
                }

                ctx.lineWidth = 1.0;
                ctx.stroke();
            }
        };

        return self;
    };


    /*
    Composite label entity
     */
    var CompositeLabelEntity = function(view, composite_label_model) {
        var self = AbstractLabelEntity(view, composite_label_model);

        self._centroid = null;
        self._component_centroids = [];
        self._OUTER_RADIUS = 8.0;
        self._INNER_RADIUS = 4.0;
        self._COMPONENT_MARKER_RADIUS = 3.0;

        self.attach = function() {
            self.update();
        };

        self.detach = function() {
        };


        self._on_mouse_over_event = function() {
            for (var i = 0; i < self.ev_mouse_in.length; i++) {
                self.ev_mouse_in[i](self);
            }
            self._view.on_entity_mouse_in(self);
        };

        self._on_mouse_out_event = function() {
            for (var i = 0; i < self.ev_mouse_out.length; i++) {
                self.ev_mouse_out[i](self);
            }
            self._view.on_entity_mouse_out(self);
        };


        var super_update = self.update;
        self.update = function() {
            super_update();
            self._component_centroids = self._compute_component_centroids();
            self._centroid = compute_centroid_of_points(self._component_centroids);
        };

        self.commit = function() {
            self._view.commit_model(self.model);
        };


        self._compute_component_centroids = function() {
            var component_centroids = [];
            for (var i = 0; i < self.model.components.length; i++) {
                var model_id = self.model.components[i];
                var entity = self._view.get_entity_for_model_id(model_id);
                var centroid = entity.compute_centroid();
                component_centroids.push(centroid);
            }
            return component_centroids;
        };

        self.compute_centroid = function() {
            return self._centroid;
        };

        self.contains_point = function(point) {
            if (self._centroid !== null) {
                var dx = point.x - self._centroid.x;
                var dy = point.y - self._centroid.y;
                return (dx * dx + dy * dy) < self._OUTER_RADIUS * self._OUTER_RADIUS;
            }
            else {
                return false;
            }
        };

        self.distance_to_point = function(point) {
            if (self._centroid !== null) {
                var dx = point.x - self._centroid.x;
                var dy = point.y - self._centroid.y;
                var dist = Math.sqrt(dx*dx + dy*dy);
                return Math.max(0.0, dist - self._OUTER_RADIUS);
            }
            else {
                return null;
            }
        };

        self.notify_model_destroyed = function(model_id) {
            var index = self.model.components.indexOf(model_id);

            if (index !== -1) {
                // Remove the model ID from the components array
                self.model.components = self.model.components.slice(0, index).concat(self.model.components.slice(index+1));
                self.update();
            }
        };


        self.paint = function(ctx, scale_x, scale_y) {
            if (self.model.components.length > 0) {
                var stroke_colour = self._selected ? [255,0,0] : [255,255,0];
                var circle_fill_colour = [255, 128, 255];
                var central_circle_fill_colour = self._view.colour_for_label_class(self.model.label_class);
                var connection_fill_colour = [255, 0, 255];
                var connection_stroke_colour = [255, 0, 255];
                if (self._hover) {
                    circle_fill_colour = lighten_colour(circle_fill_colour, 0.4);
                    central_circle_fill_colour = lighten_colour(central_circle_fill_colour, 0.4);
                    connection_fill_colour = lighten_colour(connection_fill_colour, 0.4);
                    connection_stroke_colour = lighten_colour(connection_stroke_colour, 0.4);
                }

                if (self._view.hide_labels) {
                    stroke_colour = rgb_to_rgba_string(stroke_colour, 0.2);
                    circle_fill_colour = null;
                    central_circle_fill_colour = rgb_to_rgba_string(central_circle_fill_colour, 0.35);
                    connection_fill_colour = null;
                    connection_stroke_colour = rgb_to_rgba_string(connection_stroke_colour, 0.2);
                }
                else {
                    stroke_colour = rgb_to_rgba_string(stroke_colour, 0.5);
                    circle_fill_colour = rgb_to_rgba_string(circle_fill_colour, 0.35);
                    central_circle_fill_colour = rgb_to_rgba_string(central_circle_fill_colour, 0.35);
                    connection_fill_colour = rgb_to_rgba_string(connection_fill_colour, 0.25);
                    connection_stroke_colour = rgb_to_rgba_string(connection_stroke_colour, 0.6);
                }

                ctx.beginPath();
                ctx.arc(scale_x(self._centroid.x), scale_y(self._centroid.y), self._OUTER_RADIUS,
                    0, 2 * Math.PI, false);
                if (circle_fill_colour !== null) {
                    ctx.fillStyle = circle_fill_colour;
                    ctx.fill();
                }
                ctx.lineWidth = 1;
                ctx.strokeStyle = connection_stroke_colour;
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(scale_x(self._centroid.x), scale_y(self._centroid.y), self._INNER_RADIUS,
                    0, 2 * Math.PI, false);
                ctx.fillStyle = central_circle_fill_colour;
                ctx.fill();
                ctx.lineWidth = 1;
                ctx.strokeStyle = stroke_colour;
                ctx.stroke();


                for (var i = 0; i < self._component_centroids.length; i++) {
                    var comp_cen = self._component_centroids[i];
                    if (comp_cen !== null) {
                        ctx.beginPath();
                        ctx.arc(scale_x(comp_cen.x), scale_y(comp_cen.y),
                            self._COMPONENT_MARKER_RADIUS, 0, 2 * Math.PI, false);
                        ctx.fillStyle = connection_fill_colour;
                        ctx.fill();
                        ctx.lineWidth = 1;
                        ctx.strokeStyle = connection_stroke_colour;
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.moveTo(scale_x(comp_cen.x), scale_y(comp_cen.y));
                        ctx.lineTo(scale_x(self._centroid.x), scale_y(self._centroid.y));
                        ctx.lineWidth = 1;
                        if (ctx.setLineDash) {
                            ctx.setLineDash([3, 3]);
                        }
                        ctx.strokeStyle = connection_stroke_colour;
                        ctx.stroke();
                        if (ctx.setLineDash) {
                            ctx.setLineDash(null);
                        }
                    }
                }
            }
        };

        return self;
    };



    /*
    Map label type to entity constructor
     */
    var label_type_to_entity_constructor = {
        'polygon': PolygonalLabelEntity,
        'composite': CompositeLabelEntity
    };


    /*
    Construct entity for given label model.
    Uses the map above to choose the appropriate constructor
     */
    var new_entity_for_model = function(view, label_model) {
        var constructor = label_type_to_entity_constructor[label_model.label_type];
        return constructor(view, label_model);
    };



    /*
    Abstract tool
     */
    var AbstractTool = function(view) {
        var self = {
            _view: view
        };

        self.on_init = function() {
        };

        self.on_shutdown = function() {
        };

        self.on_switch_in = function(pos) {
        };

        self.on_switch_out = function(pos) {
        };

        self.on_left_click = function(pos, event) {
        };

        self.on_cancel = function(pos) {
        };

        self.on_button_down = function(pos, event) {
        };

        self.on_button_up = function(pos, event) {
        };

        self.on_move = function(pos) {
        };

        self.on_drag = function(pos) {
        };

        self.on_wheel = function(pos, wheelDeltaX, wheelDeltaY) {
        };

        self.on_key_down = function(event) {
        };

        self.on_entity_mouse_in = function(entity) {
        };

        self.on_entity_mouse_out = function(entity) {
        };

        self.paint = function(ctx, scale_x, scale_y) {
        };

        self.queue_redraw = function() {
            self._view.queue_redraw();
        };

        return self;
    };


    /*
    Select entity tool
     */
    var SelectEntityTool = function(view) {
        var self = AbstractTool(view);

        self._current_entity = null;

        self.on_init = function() {
            self._current_entity = null;
        };

        self.on_shutdown = function() {
            // Remove any hover
            if (self._current_entity !== null) {
                self._current_entity.hover(false);
                self._current_entity = null;
            }
        };

        self.on_switch_in = function(pos) {
            var entities = self._view.get_entities_under_point(pos);
            self._set_current_entity(entities.length > 0 ? entities[entities.length - 1] : null);
        };

        self.on_switch_out = function(pos) {
            self._set_current_entity(null);
        };


        self.on_move = function(pos, event) {
            var entities = self._view.get_entities_under_point(pos);
            self._set_current_entity(entities.length > 0 ? entities[entities.length - 1] : null);
            return true;
        };

        self._set_current_entity = function(entity) {
            if (entity !== self._current_entity) {
                if (self._current_entity !== null) {
                    self._current_entity.hover(false);
                }
                self._current_entity = entity;
                if (self._current_entity !== null) {
                    self._current_entity.hover(true);
                }
            }
        };

        self.on_left_click = function(pos, event) {
            var entity = self._current_entity;
            if (entity !== null) {
                self._view.select_entity(entity, event.shiftKey, true);
            }
            else {
                if (!event.shiftKey) {
                    self._view.unselect_all_entities();
                }
            }
        };

        return self;
    };


    /*
    Brush select entity tool
     */
    var BrushSelectEntityTool = function(view) {
        var self = AbstractTool(view);

        self._highlighted_entities = [];
        self._brush_radius = 10.0;
        self._brush_centre = null;

        self.on_init = function() {
            self._brush_centre = null;
            self._highlighted_entities = [];
        };

        self.on_shutdown = function() {
            self._brush_centre = null;
        };


        self._get_entities_in_range = function(point) {
            var in_range = [];
            var entities = self._view.get_entities();
            for (var i = 0; i < entities.length; i++) {
                var entity = entities[i];
                var dist = entity.distance_to_point(point);
                if (dist !== null) {
                    if (dist <= self._brush_radius) {
                        in_range.push(entity);
                    }
                }
            }
            return in_range;
        };

        self._highlight_entities = function(entities) {
            // Remove any hover
            for (var i = 0; i < self._highlighted_entities.length; i++) {
                self._highlighted_entities[i].hover(false);
            }

            self._highlighted_entities = entities;

            // Add hover
            for (var i = 0; i < self._highlighted_entities.length; i++) {
                self._highlighted_entities[i].hover(true);
            }
        };


        self.on_button_down = function(pos, event) {
            var entities = self._get_entities_in_range(pos);
            for (var i = 0; i < entities.length; i++) {
                self._view.select_entity(entities[i], event.shiftKey || i > 0, false);
            }
            return true;
        };

        self.on_button_up = function(pos, event) {
            self._highlight_entities(self._get_entities_in_range(pos));
            return true;
        };

        self.on_move = function(pos, event) {
            self._highlight_entities(self._get_entities_in_range(pos));
            self._brush_centre = self._view.get_mouse_pos_screen_space();
            self.queue_redraw();
            return true;
        };

        self.on_drag = function(pos, event) {
            var entities = self._get_entities_in_range(pos);
            for (var i = 0; i < entities.length; i++) {
                self._view.select_entity(entities[i], true, false);
            }
            self._brush_centre = self._view.get_mouse_pos_screen_space();
            self.queue_redraw();
            return true;
        };

        self.on_wheel = function(pos, wheelDeltaX, wheelDeltaY) {
            self._brush_radius += wheelDeltaY * 0.1;
            self._brush_radius = Math.max(self._brush_radius, 1.0);
            self.queue_redraw();
            return true;
        };

        self.on_key_down = function(event) {
            var changed = false;
            if (event.keyCode == 219) {
                self._brush_radius -= 2.0;
                changed = true;
            }
            else if (event.keyCode == 221) {
                self._brush_radius += 2.0;
                changed = true;
            }
            if (changed) {
                self._brush_radius = Math.max(self._brush_radius, 1.0);
                self.queue_redraw();
                return true;
            }
        };

        self.on_switch_in = function(pos) {
            self._highlight_entities(self._get_entities_in_range(pos));
            self._brush_centre = self._view.get_mouse_pos_screen_space();
            self.queue_redraw();
        };

        self.on_switch_out = function(pos) {
            self._highlight_entities([]);
            self._brush_centre = null;
            self.queue_redraw();
        };

        self.paint = function(ctx, scale_x, scale_y) {
            if (self._brush_centre !== null) {
                ctx.beginPath();
                ctx.arc(self._brush_centre.x, self._brush_centre.y, self._brush_radius, 0, 2 * Math.PI, false);
                ctx.fillStyle = 'rgba(128,0,0,0.05)';
                ctx.fill();
                ctx.lineWidth = 1;
                ctx.strokeStyle = '#ff0000';
                ctx.stroke();
            }
        };

        return self;
    };


    /*
    Draw polygon tool
     */
    var DrawPolygonTool = function(view, entity) {
        var self = AbstractTool(view);

        self.entity = entity;

        self.on_init = function() {
        };

        self.on_shutdown = function() {
        };

        self.on_switch_in = function(pos) {
            if (self.entity !== null) {
                self.add_point(pos);
            }
        };

        self.on_switch_out = function(pos) {
            if (self.entity !== null) {
                self.remove_last_point();
            }
        };

        self.on_cancel = function(pos) {
            if (self.entity !== null) {
                self.remove_last_point();

                var vertices = self.get_vertices();
                if (vertices.length == 1) {
                    self.destroy_entity();
                }
                else {
                    self.entity.commit();
                    self.entity = null;
                }
            }
            else {
                self._view.unselect_all_entities();
                self._view.set_current_tool(SelectEntityTool(self._view));
            }
        };

        self.on_left_click = function(pos, event) {
            self.add_point(pos);
        };

        self.on_move = function(pos) {
            self.update_last_point(pos);
        };



        self.create_entity = function() {
            var model = PolygonalLabelModel();
            var entity = PolygonalLabelEntity(self._view, model);
            self.entity = entity;
            self._view.add_entity(entity, false);
            self._view.select_entity(entity, false, false);
        };

        self.destroy_entity = function() {
            self._view.remove_entity(self.entity, false);
            self.entity = null;
        };

        self.get_vertices = function() {
            return self.entity !== null  ?  self.entity.model.vertices  :  null;
        };

        self.update_poly = function() {
            if (self.entity !== null) {
                self.entity.update();
            }
        };

        self.add_point = function(pos) {
            var entity_is_new = false;
            if (self.entity === null) {
                self.create_entity();
                entity_is_new = true;
            }
            var vertices = self.get_vertices();

            if (entity_is_new) {
                // Add a duplicate vertex; this second vertex will follow the mouse
                vertices.push(pos);
            }
            vertices.push(pos);
            self.update_poly();
        };

        self.update_last_point = function(pos) {
            var vertices = self.get_vertices();
            if (vertices !== null) {
                vertices[vertices.length - 1] = pos;
                self.update_poly();
            }
        };

        self.remove_last_point = function() {
            var vertices = self.get_vertices();

            if (vertices !== null) {
                if (vertices.length > 0) {
                    vertices.splice(vertices.length - 1, 1);
                    self.update_poly();
                }

                if (vertices.length === 0) {
                    self.destroy_entity();
                }
            }
        };

        return self;
    };



    /*
    Labelling tool view; links to the server side data structures
     */
    var LabellingToolSelf = {};


    var ensure_flag_exists = function(x, flag_name, default_value) {
        var v = x[flag_name];
        if (v === undefined) {
            x[flag_name] = default_value;
        }
        return x[flag_name];
    };


    LabellingToolSelf.initialise = function(element, label_classes, tool_width, tool_height,
                                            image_ids, initial_image_id, requestImageCallback, sendLabelHeaderFn, config) {
        config = config || {};
        LabellingToolSelf._config = config;

        config.tools = config.tools || {};
        ensure_flag_exists(config.tools, 'imageSelector', true);
        ensure_flag_exists(config.tools, 'labelClassSelector', true);
        ensure_flag_exists(config.tools, 'brushSelect', true);
        ensure_flag_exists(config.tools, 'drawPolyLabel', true);
        ensure_flag_exists(config.tools, 'compositeLabel', true);
        ensure_flag_exists(config.tools, 'deleteLabel', true);


        // Model
        LabellingToolSelf._label_header = {};
        // Entity list
        LabellingToolSelf.entities = [];
        // Active tool
        LabellingToolSelf.$tool = null;
        // Selected entity
        LabellingToolSelf.$selected_entities = [];
        // Classes
        LabellingToolSelf.$label_classes = label_classes;
        // Hide labels
        LabellingToolSelf.hide_labels = false;
        // Button state
        LabellingToolSelf._button_down = false;
        // Redraw queue timeout
        LabellingToolSelf._redraw_timeout = null;

        // Label model object table
        LabellingToolSelf._label_model_obj_table = ObjectIDTable();
        // Label model object ID to entity
        LabellingToolSelf._label_model_id_to_entity = {};

        // Labelling tool dimensions
        LabellingToolSelf._tool_width = tool_width;
        LabellingToolSelf._tool_height = tool_height;

        // List of Image IDs
        LabellingToolSelf._image_ids = image_ids;

        // Number of images in dataset
        LabellingToolSelf._num_images = image_ids.length;

        // Image dimensions
        LabellingToolSelf._image_width = 0;
        LabellingToolSelf._image_height = 0;

        // Data request callback; labelling tool will call this when it needs a new image to show
        LabellingToolSelf._requestImageCallback = requestImageCallback;
        // Send data callback; labelling tool will call this when it wants to commit data to the backend in response
        // to user action
        LabellingToolSelf._sendLabelHeaderFn = sendLabelHeaderFn;


        var toolbar_width = 220;
        LabellingToolSelf._labelling_area_width = LabellingToolSelf._tool_width - toolbar_width;
        var labelling_area_x_pos = toolbar_width + 10;


        // A <div> element that surrounds the labelling tool
        LabellingToolSelf._div = $('<div style="border: 1px solid gray; width: ' + LabellingToolSelf._tool_width + 'px;"/>')
            .appendTo(element);

        var toolbar_container = $('<div style="position: relative;">').appendTo(LabellingToolSelf._div);

        LabellingToolSelf._toolbar = $('<div style="position: absolute; width: ' + toolbar_width + 'px; padding: 4px; display: inline-block; background: #d0d0d0; border: 1px solid #a0a0a0;"/>').appendTo(toolbar_container);
        LabellingToolSelf._labelling_area = $('<div style="width:' + LabellingToolSelf._labelling_area_width + 'px; margin-left: ' + labelling_area_x_pos + 'px"/>').appendTo(LabellingToolSelf._div);


        /*
         *
         *
         * TOOLBAR CONTENTS
         *
         *
         */

        //
        // IMAGE SELECTOR
        //

        $('<p style="background: #b0b0b0;">Current image</p>').appendTo(LabellingToolSelf._toolbar);

        if (config.tools.imageSelector) {
            var _change_image = function (image_id) {
                LabellingToolSelf._requestImageCallback(image_id);
            };

            var _increment_image_index = function (offset) {
                var image_id = LabellingToolSelf._get_current_image_id();
                var index = LabellingToolSelf._image_id_to_index(image_id) + offset;
                _change_image(LabellingToolSelf._image_index_to_id(index));
            };

            LabellingToolSelf._image_index_input = $('<input type="text" style="width: 30px; vertical-align: middle;" name="image_index"/>').appendTo(LabellingToolSelf._toolbar);
            LabellingToolSelf._image_index_input.on('change', function () {
                var index_str = LabellingToolSelf._image_index_input.val();
                var index = parseInt(index_str) - 1;
                var image_id = LabellingToolSelf._image_index_to_id(index);
                _change_image(image_id);
            });
            $('<span>' + '/' + LabellingToolSelf._num_images + '</span>').appendTo(LabellingToolSelf._toolbar);


            $('<br/>').appendTo(LabellingToolSelf._toolbar);
            LabellingToolSelf._prev_image_button = $('<button>Prev image</button>').appendTo(LabellingToolSelf._toolbar);
            LabellingToolSelf._prev_image_button.button({
                text: false,
                icons: {primary: "ui-icon-seek-prev"}
            }).click(function (event) {
                _increment_image_index(-1);
                event.preventDefault();
            });

            LabellingToolSelf._next_image_button = $('<button>Next image</button>').appendTo(LabellingToolSelf._toolbar);
            LabellingToolSelf._next_image_button.button({
                text: false,
                icons: {primary: "ui-icon-seek-next"}
            }).click(function (event) {
                _increment_image_index(1);
                event.preventDefault();
            });
        }

        $('<br/>').appendTo(LabellingToolSelf._toolbar);
        LabellingToolSelf._complete_checkbox = $('<input type="checkbox">Finished</input>').appendTo(LabellingToolSelf._toolbar);
        LabellingToolSelf._complete_checkbox.change(function(event, ui) {
            var value = event.target.checked;
            LabellingToolSelf._label_header.complete = value;
            LabellingToolSelf.push_label_data();
        });




        //
        // LABEL CLASS SELECTOR AND HIDE LABELS
        //

        $('<p style="background: #b0b0b0;">Labels</p>').appendTo(LabellingToolSelf._toolbar);

        if (config.tools.labelClassSelector) {
            LabellingToolSelf._label_class_selector_menu = $('<select name="label_class_selector"/>').appendTo(LabellingToolSelf._toolbar);
            for (var i = 0; i < LabellingToolSelf.$label_classes.length; i++) {
                var cls = LabellingToolSelf.$label_classes[i];
                $('<option value="' + cls.name + '">' + cls.human_name + '</option>').appendTo(LabellingToolSelf._label_class_selector_menu);
            }
            $('<option value="__unclassified" selected="false">UNCLASSIFIED</option>').appendTo(LabellingToolSelf._label_class_selector_menu);
            LabellingToolSelf._label_class_selector_menu.change(function (event, ui) {
                var label_class_name = event.target.value;
                if (label_class_name == '__unclassified') {
                    label_class_name = null;
                }
                for (var i = 0; i < LabellingToolSelf.$selected_entities.length; i++) {
                    LabellingToolSelf.$selected_entities[i].set_label_class(label_class_name);
                }
            });
        }

        $('<br/>').appendTo(LabellingToolSelf._toolbar);
        LabellingToolSelf._hide_labels_checkbox = $('<input type="checkbox">Hide labels</input>').appendTo(LabellingToolSelf._toolbar);
        LabellingToolSelf._hide_labels_checkbox.change(function(event, ui) {
            var value = event.target.checked;
            LabellingToolSelf.hide_labels = value;

            for (var i = 0; i < LabellingToolSelf.entities.length; i++) {
                LabellingToolSelf.entities[i].notify_hide_labels_change(value);
            }
        });





        //
        // SELECT, DRAW POLY, COMPOSITE, DELETE
        //

        $('<p style="background: #b0b0b0;">Tools</p>').appendTo(LabellingToolSelf._toolbar);
        LabellingToolSelf._select_button = $('<button>Select</button>').appendTo(LabellingToolSelf._toolbar);
        LabellingToolSelf._select_button.button().click(function(event) {
            LabellingToolSelf.set_current_tool(SelectEntityTool(LabellingToolSelf));
            event.preventDefault();
        });

        if (config.tools.brushSelect) {
            LabellingToolSelf._brush_select_button = $('<button>Brush select</button>').appendTo(LabellingToolSelf._toolbar);
            LabellingToolSelf._brush_select_button.button().click(function (event) {
                LabellingToolSelf.set_current_tool(BrushSelectEntityTool(LabellingToolSelf));
                event.preventDefault();
            });
        }

        if (config.tools.drawPolyLabel) {
            LabellingToolSelf._draw_polygon_button = $('<button>Draw poly</button>').appendTo(LabellingToolSelf._toolbar);
            LabellingToolSelf._draw_polygon_button.button().click(function (event) {
                var current = LabellingToolSelf.get_selected_entity();
                LabellingToolSelf.set_current_tool(DrawPolygonTool(LabellingToolSelf, current));
                event.preventDefault();
            });
        }

        if (config.tools.compositeLabel) {
            LabellingToolSelf._composite_button = $('<button>Composite</button>').appendTo(LabellingToolSelf._toolbar);
            LabellingToolSelf._composite_button.button().click(function (event) {
                var N = LabellingToolSelf.$selected_entities.length;

                if (N > 0) {
                    var model = CompositeLabelModel();
                    var entity = CompositeLabelEntity(LabellingToolSelf, model);

                    for (var i = 0; i < LabellingToolSelf.$selected_entities.length; i++) {
                        model.components.push(LabellingToolSelf.$selected_entities[i].model.object_id);
                    }

                    LabellingToolSelf.add_entity(entity, true);
                    LabellingToolSelf.select_entity(entity, false, false);
                }

                event.preventDefault();
            });
        }

        if (config.tools.deleteLabel) {
            LabellingToolSelf._delete_label_button = $('<button>Delete</button>').appendTo(LabellingToolSelf._toolbar);
            LabellingToolSelf._delete_label_button.button({
                text: false,
                icons: {primary: "ui-icon-trash"}
            }).click(function (event) {
                if (!LabellingToolSelf._confirm_delete_visible) {
                    var cancel_button = $('<button>Cancel</button>').appendTo(LabellingToolSelf._confirm_delete);
                    var confirm_button = $('<button>Confirm delete</button>').appendTo(LabellingToolSelf._confirm_delete);

                    var remove_confirm_ui = function () {
                        cancel_button.remove();
                        confirm_button.remove();
                        LabellingToolSelf._confirm_delete_visible = false;
                    };

                    cancel_button.button().click(function (event) {
                        remove_confirm_ui();
                        event.preventDefault();
                    });

                    confirm_button.button().click(function (event) {
                        var entities_to_remove = LabellingToolSelf.$selected_entities.slice();

                        for (var i = 0; i < entities_to_remove.length; i++) {
                            LabellingToolSelf.remove_entity(entities_to_remove[i], true);
                        }

                        remove_confirm_ui();
                        event.preventDefault();
                    });

                    LabellingToolSelf._confirm_delete_visible = true;
                }

                event.preventDefault();
            });

            LabellingToolSelf._confirm_delete = $('<span/>').appendTo(LabellingToolSelf._toolbar);
            LabellingToolSelf._confirm_delete_visible = false;
        }




        /*
         *
         * LABELLING AREA
         *
         */

        // Zoom callback
        LabellingToolSelf._zoom_xlat = [0.0, 0.0];
        LabellingToolSelf._zoom_scale = 1.0;
        function zoomed() {
            LabellingToolSelf._zoom_xlat = d3.event.translate;
            LabellingToolSelf._zoom_scale = d3.event.scale;
            LabellingToolSelf.canvasContext.clearRect(0, 0,
                    LabellingToolSelf._labelling_area_width, LabellingToolSelf._tool_height);
            LabellingToolSelf._paint();
        }

        // Disable context menu so we can use right-click
        LabellingToolSelf._labelling_area[0].oncontextmenu = function() {
            return false;
        };

        var x = d3.scale.linear()
            .domain([0, LabellingToolSelf._labelling_area_width])
            .range([0, LabellingToolSelf._labelling_area_width]);
        LabellingToolSelf.$scale_x = x;

        //var y = d3.scale.linear()
        //    .domain([0, LabellingToolSelf._tool_height])
        //    .range([LabellingToolSelf._tool_height, 0]);
        var y = d3.scale.linear()
            .domain([0, LabellingToolSelf._tool_height])
            .range([0, LabellingToolSelf._tool_height]);
        LabellingToolSelf.$scale_y = y;

        LabellingToolSelf.$scale_factor = 1.0;

        var d3LabellingArea = d3.select(LabellingToolSelf._labelling_area[0]);
        LabellingToolSelf.$canvas = d3LabellingArea.append("canvas")
            .attr("width", LabellingToolSelf._labelling_area_width)
            .attr("height", LabellingToolSelf._tool_height)
            .call(d3.behavior.zoom().x(x).y(y).on("zoom", zoomed));
        LabellingToolSelf.canvasContext = LabellingToolSelf.$canvas.node().getContext("2d");
        var canvas = LabellingToolSelf.$canvas;


        // Create image object
        LabellingToolSelf._imageObject = new Image();
        LabellingToolSelf._imageObject.onload = function() {
            LabellingToolSelf._paint();
        };


        // Flag that indicates if the mouse pointer is within the tool area
        LabellingToolSelf._mouse_within = false;
        LabellingToolSelf._last_mouse_pos = null;


        //
        // Set up event handlers
        //

        // Click
        canvas.on("click", function() {
            if (d3.event.button === 0) {
                // Left click; send to tool
                var handled = false;
                if (LabellingToolSelf.$tool !== null) {
                    handled = LabellingToolSelf.$tool.on_left_click(LabellingToolSelf.get_mouse_pos_world_space(), d3.event);
                }

                if (handled) {
                    d3.event.stopPropagation();
                }
            }
        });

        // Button press
        canvas.on("mousedown", function() {
            var handled = false;
            if (d3.event.button === 0) {
                // Left button down
                LabellingToolSelf._button_down = true;
                if (LabellingToolSelf.$tool !== null) {
                    handled = LabellingToolSelf.$tool.on_button_down(LabellingToolSelf.get_mouse_pos_world_space(), d3.event);
                }
            }
            else if (d3.event.button === 2) {
                // Right click; on_cancel current tool
                if (LabellingToolSelf.$tool !== null) {
                    handled = LabellingToolSelf.$tool.on_cancel(LabellingToolSelf.get_mouse_pos_world_space());
                }
            }
            if (handled) {
                d3.event.stopPropagation();
            }
        });

        // Button press
        canvas.on("mouseup", function() {
            var handled = false;
            if (d3.event.button === 0) {
                // Left buton up
                LabellingToolSelf._button_down = false;
                if (LabellingToolSelf.$tool !== null) {
                    handled = LabellingToolSelf.$tool.on_button_up(LabellingToolSelf.get_mouse_pos_world_space(), d3.event);
                }
            }
            if (handled) {
                d3.event.stopPropagation();
            }
        });

        // Mouse on_move
        canvas.on("mousemove", function() {
            var handled = false;
            LabellingToolSelf._last_mouse_pos = LabellingToolSelf.get_mouse_pos_world_space();
            if (LabellingToolSelf._button_down) {
                if (LabellingToolSelf.$tool !== null) {
                    handled = LabellingToolSelf.$tool.on_drag(LabellingToolSelf._last_mouse_pos);
                }
            }
            else {
                if (!LabellingToolSelf._mouse_within) {
                    LabellingToolSelf._init_key_handlers();

                    // Entered tool area; invoke tool.on_switch_in()
                    if (LabellingToolSelf.$tool !== null) {
                        handled = LabellingToolSelf.$tool.on_switch_in(LabellingToolSelf._last_mouse_pos);
                    }

                    LabellingToolSelf._mouse_within = true;
                }
                else {
                    // Send mouse on_move event to tool
                    if (LabellingToolSelf.$tool !== null) {
                        handled = LabellingToolSelf.$tool.on_move(LabellingToolSelf._last_mouse_pos);
                    }
                }
            }
            if (handled) {
                d3.event.stopPropagation();
            }
        });

        // Mouse wheel
        canvas.on("mousewheel", function() {
            var handled = false;
            LabellingToolSelf._last_mouse_pos = LabellingToolSelf.get_mouse_pos_world_space();
            if (d3.event.ctrlKey || d3.event.shiftKey || d3.event.altKey) {
                if (LabellingToolSelf.$tool !== null) {
                    handled = LabellingToolSelf.$tool.on_wheel(LabellingToolSelf._last_mouse_pos,
                                                               d3.event.wheelDeltaX, d3.event.wheelDeltaY);
                }
            }
            if (handled) {
                d3.event.stopPropagation();
            }
        });


        var on_mouse_out = function(pos, width, height) {
            if (LabellingToolSelf._mouse_within) {
                if (pos.x < 0.0 || pos.x > width || pos.y < 0.0 || pos.y > height) {
                    // The pointer is outside the bounds of the tool, as opposed to entering another element within the bounds of the tool, e.g. a polygon
                    // invoke tool.on_switch_out()
                    var handled = false;
                    if (LabellingToolSelf.$tool !== null) {
                        handled = LabellingToolSelf.$tool.on_switch_out(pos);
                    }

                    if (handled) {
                        d3.event.stopPropagation();
                    }

                    LabellingToolSelf._mouse_within = false;
                    LabellingToolSelf._last_mouse_pos = null;
                    LabellingToolSelf._shutdown_key_handlers();
                }
            }
        };

        // Mouse leave
        LabellingToolSelf.$canvas.on("mouseout", function() {
            var width = LabellingToolSelf.$canvas.attr("width");
            var height = LabellingToolSelf.$canvas.attr("height");
            on_mouse_out(LabellingToolSelf.get_mouse_pos_world_space(), width, height);
        });


        // Global key handler
        if (!__labelling_tool_key_handler.connected) {
            d3.select("body").on("keydown", function () {
                if (__labelling_tool_key_handler.handler !== null) {
                    var handled = __labelling_tool_key_handler.handler(d3.event);
                    if (handled) {
                        d3.event.stopPropagation();
                    }
                }
            });
            __labelling_tool_key_handler.connected = true;
        }


        // Create entities for the pre-existing labels
        LabellingToolSelf._requestImageCallback(initial_image_id)
    };


    LabellingToolSelf._paint = function() {
        LabellingToolSelf.canvasContext.clearRect(0, 0,
            LabellingToolSelf._labelling_area_width,  LabellingToolSelf._tool_height);

        LabellingToolSelf.canvasContext.save();
        LabellingToolSelf.canvasContext.translate(LabellingToolSelf._zoom_xlat[0], LabellingToolSelf._zoom_xlat[1]);
        LabellingToolSelf.canvasContext.scale(LabellingToolSelf._zoom_scale, LabellingToolSelf._zoom_scale);

        LabellingToolSelf.canvasContext.drawImage(LabellingToolSelf._imageObject, 0.0, 0.0);
        LabellingToolSelf.canvasContext.restore();

        for (var i = 0; i < LabellingToolSelf.entities.length; i++) {
            LabellingToolSelf.entities[i].paint(LabellingToolSelf.canvasContext,
                LabellingToolSelf.$scale_x, LabellingToolSelf.$scale_y);
        }

        if (LabellingToolSelf.$tool !== null) {
            LabellingToolSelf.$tool.paint(LabellingToolSelf.canvasContext,
                LabellingToolSelf.$scale_x, LabellingToolSelf.$scale_y);
        }

        console.log(LabellingToolSelf.canvasContext.getLineDash());
    };


    LabellingToolSelf.queue_redraw = function() {
        if (LabellingToolSelf._redraw_timeout === null) {
            LabellingToolSelf._redraw_timeout = setTimeout(function() {
                LabellingToolSelf._redraw_timeout = null;
                LabellingToolSelf._paint();
            }, 0);
        }
    };


    LabellingToolSelf._image_id_to_index = function(image_id) {
        var image_index = LabellingToolSelf._image_ids.indexOf(image_id);
        if (image_index === -1) {
            console.log("Image ID " + image_id + " not found");
            image_index = 0;
        }
        return image_index;
    };

    LabellingToolSelf._image_index_to_id = function(index) {
        var clampedIndex = Math.max(Math.min(index, LabellingToolSelf._image_ids.length - 1), 0);
        console.log("index=" + index + ", clampedIndex="+clampedIndex);
        return LabellingToolSelf._image_ids[clampedIndex];
    };

    LabellingToolSelf._update_image_index_input = function(image_id) {
        var image_index = LabellingToolSelf._image_id_to_index(image_id);

        LabellingToolSelf._image_index_input.val((image_index+1).toString());
    };

    LabellingToolSelf._get_current_image_id = function() {
        if (LabellingToolSelf._label_header !== null  &&  LabellingToolSelf._label_header !== undefined) {
            return LabellingToolSelf._label_header.image_id;
        }
        else {
            return null;
        }
    };

    LabellingToolSelf.setImage = function(image_data) {
        // Remove all entities
        while (LabellingToolSelf.entities.length > 0) {
            LabellingToolSelf.unregister_entity_by_index(LabellingToolSelf.entities.length-1);
        }

        // Update the image SVG element
        LabellingToolSelf._imageObject.src = image_data.href;

        LabellingToolSelf._image_width = image_data.width;
        LabellingToolSelf._image_height = image_data.height;

        // Update the labels
        LabellingToolSelf._label_header = image_data.label_header;
        var labels = LabellingToolSelf.get_labels_in_header();

        // Set up the ID counter; ensure that it's value is 1 above the maximum label ID in use
        LabellingToolSelf._label_model_obj_table = ObjectIDTable();
        LabellingToolSelf._label_model_obj_table.register_objects(labels);

        for (var i = 0; i < labels.length; i++) {
            var label = labels[i];
            var entity = new_entity_for_model(LabellingToolSelf, label);
            LabellingToolSelf.register_entity(entity);
        }

        LabellingToolSelf._complete_checkbox[0].checked = LabellingToolSelf._label_header.complete;

        LabellingToolSelf._update_image_index_input(LabellingToolSelf._label_header.image_id);


        LabellingToolSelf.set_current_tool(SelectEntityTool(LabellingToolSelf));

        console.log(LabellingToolSelf);
    };

    LabellingToolSelf.get_labels_in_header = function() {
        var labels = LabellingToolSelf._label_header.labels;
        if (labels === null) {
            labels = [];
        }
        return labels;
    };






    /*
    Entity mouse in event
     */
    LabellingToolSelf.on_entity_mouse_in = function(entity) {
        if (LabellingToolSelf.$tool !== null) {
            LabellingToolSelf.$tool.on_entity_mouse_in(entity);
        }
    };

    LabellingToolSelf.on_entity_mouse_out = function(entity) {
        if (LabellingToolSelf.$tool !== null) {
            LabellingToolSelf.$tool.on_entity_mouse_out(entity);
        }
    };

    LabellingToolSelf.get_entities_under_point = function(pos) {
        var entities = [];
        for (var i = 0; i < LabellingToolSelf.entities.length; i++) {
            var entity = LabellingToolSelf.entities[i];
            if (entity.contains_point(pos)) {
                entities.push(entity);
            }
        }
        return entities;
    };



    /*
    Get colour for a given label class
     */
    LabellingToolSelf.index_for_label_class = function(label_class) {
        if (label_class != null) {
            for (var i = 0; i < LabellingToolSelf.$label_classes.length; i++) {
                var cls = LabellingToolSelf.$label_classes[i];

                if (cls.name === label_class) {
                    return i;
                }
            }
        }

        // Default
        return -1;
    };

    LabellingToolSelf.colour_for_label_class = function(label_class) {
        var index = LabellingToolSelf.index_for_label_class(label_class);
        if (index !== -1) {
            return LabellingToolSelf.$label_classes[index].colour;
        }
        else {
            // Default
            return [0, 0, 0];
        }
    };

    LabellingToolSelf._update_label_class_menu = function(label_class) {
        if (label_class === null) {
            label_class = '__unclassified';
        }

        LabellingToolSelf._label_class_selector_menu.children('option').each(function() {
            this.selected = (this.value == label_class);
        });
    };



    /*
    Set the current tool; switch the old one out and a new one in
     */
    LabellingToolSelf.set_current_tool = function(tool) {
        if (LabellingToolSelf.$tool !== null) {
            if (LabellingToolSelf._mouse_within) {
                LabellingToolSelf.$tool.on_switch_out(LabellingToolSelf._last_mouse_pos);
            }
            LabellingToolSelf.$tool.on_shutdown();
        }

        LabellingToolSelf.$tool = tool;

        if (LabellingToolSelf.$tool !== null) {
            LabellingToolSelf.$tool.on_init();
            if (LabellingToolSelf._mouse_within) {
                LabellingToolSelf.$tool.on_switch_in(LabellingToolSelf._last_mouse_pos);
            }
        }
    };


    /*
    Select an entity
     */
    LabellingToolSelf.select_entity = function(entity, multi_select, invert) {
        multi_select = multi_select === undefined  ?  false  :  multi_select;

        if (multi_select) {
            var index = LabellingToolSelf.$selected_entities.indexOf(entity);
            var changed = false;

            if (invert) {
                if (index === -1) {
                    // Add
                    LabellingToolSelf.$selected_entities.push(entity);
                    entity.select(true);
                    changed = true;
                }
                else {
                    // Remove
                    LabellingToolSelf.$selected_entities.splice(index, 1);
                    entity.select(false);
                    changed = true;
                }
            }
            else {
                if (index === -1) {
                    // Add
                    LabellingToolSelf.$selected_entities.push(entity);
                    entity.select(true);
                    changed = true;
                }
            }

            if (changed) {
                if (LabellingToolSelf.$selected_entities.length === 1) {
                    LabellingToolSelf._update_label_class_menu(LabellingToolSelf.$selected_entities[0].get_label_class());
                }
                else {
                    LabellingToolSelf._update_label_class_menu(null);
                }
            }
        }
        else {
            var prev_entity = LabellingToolSelf.get_selected_entity();

            if (prev_entity !== entity) {
                for (var i = 0; i < LabellingToolSelf.$selected_entities.length; i++) {
                    LabellingToolSelf.$selected_entities[i].select(false);
                }
                LabellingToolSelf.$selected_entities = [entity];
                entity.select(true);
            }

            LabellingToolSelf._update_label_class_menu(entity.get_label_class());
        }
    };


    /*
    Unselect all entities
     */
    LabellingToolSelf.unselect_all_entities = function() {
        for (var i = 0; i < LabellingToolSelf.$selected_entities.length; i++) {
            LabellingToolSelf.$selected_entities[i].select(false);
        }
        LabellingToolSelf.$selected_entities = [];
        LabellingToolSelf._update_label_class_menu(null);
    };


    /*
    Get uniquely selected entity
     */
    LabellingToolSelf.get_selected_entity = function() {
        return LabellingToolSelf.$selected_entities.length == 1  ?  LabellingToolSelf.$selected_entities[0]  :  null;
    };

    /*
    Get all entities
     */
    LabellingToolSelf.get_entities = function() {
        return LabellingToolSelf.entities;
    };



    /*
    Register entity
     */
    LabellingToolSelf.register_entity = function(entity) {
        LabellingToolSelf._label_model_obj_table.register(entity.model);
        LabellingToolSelf.entities.push(entity);
        LabellingToolSelf._label_model_id_to_entity[entity.model.object_id] = entity;
        entity.attach();
    };

    /*
    Unregister entity by index
     */
    LabellingToolSelf.unregister_entity_by_index = function(index) {
        var entity = LabellingToolSelf.entities[index];

        // Notify all models of the destruction of this model
        for (var i = 0; i < LabellingToolSelf.entities.length; i++) {
            if (i !== index) {
                LabellingToolSelf.entities[i].notify_model_destroyed(entity.model);
            }
        }

        // Unregister in the ID to object table
        LabellingToolSelf._label_model_obj_table.unregister(entity.model);
        delete LabellingToolSelf._label_model_id_to_entity[entity.model.object_id];


        // Remove from selection if present
        var index_in_selection = LabellingToolSelf.$selected_entities.indexOf(entity);
        if (index_in_selection !== -1) {
            entity.select(false);
            LabellingToolSelf.$selected_entities.splice(index_in_selection, 1);
        }

        entity.detach();
        // Remove
        LabellingToolSelf.entities.splice(index, 1);
    };


    /*
    Get entity for model ID
     */
    LabellingToolSelf.get_entity_for_model_id = function(model_id) {
        return LabellingToolSelf._label_model_id_to_entity[model_id];
    };

    /*
    Get entity for model
     */
    LabellingToolSelf.get_entity_for_model = function(model) {
        return LabellingToolSelf._label_model_id_to_entity[model.object_id];
    };



    /*
    Add entity:
    register the entity and add its label to the tool data model
     */
    LabellingToolSelf.add_entity = function(entity, commit) {
        LabellingToolSelf.register_entity(entity);

        var labels = LabellingToolSelf.get_labels_in_header();
        labels = labels.concat([entity.model]);
        LabellingToolSelf._label_header = replace_label_header_labels(LabellingToolSelf._label_header, labels);

        if (commit) {
            LabellingToolSelf.push_label_data();
        }
    };

    /*
    Remove entity
    unregister the entity and remove its label from the tool data model
     */
    LabellingToolSelf.remove_entity = function(entity, commit) {
        // Find the entity's index in the array
        var index = LabellingToolSelf.entities.indexOf(entity);

        if (index !== -1) {
            // Unregister the entity
            LabellingToolSelf.unregister_entity_by_index(index);

            // Get the label model
            var labels = LabellingToolSelf.get_labels_in_header();

            // Remove the model from the label model array
            labels = labels.slice(0, index).concat(labels.slice(index+1));
            // Replace the labels in the label header
            LabellingToolSelf._label_header = replace_label_header_labels(LabellingToolSelf._label_header, labels);

            if (commit) {
                // Commit changes
                LabellingToolSelf.push_label_data();
            }
        }
    };

    /*
    Commit model
    invoke when a model is modified
    inserts the model into the tool data model and ensures that the relevant change events get send over
     */
    LabellingToolSelf.commit_model = function(model) {
        var labels = LabellingToolSelf.get_labels_in_header();
        var index = labels.indexOf(model);

        if (index !== -1) {
            LabellingToolSelf.push_label_data();
        }
    };

    LabellingToolSelf.push_label_data = function() {
        LabellingToolSelf._sendLabelHeaderFn(LabellingToolSelf._label_header);
    };

    // Function for getting the current mouse position
    LabellingToolSelf.get_mouse_pos_screen_space = function() {
        var pos = d3.mouse(LabellingToolSelf.$canvas[0][0]);
        return {x: pos[0], y: pos[1]};
    };

    LabellingToolSelf.get_mouse_pos_world_space = function() {
        var pos = d3.mouse(LabellingToolSelf.$canvas[0][0]);
        return {x: LabellingToolSelf.$scale_x.invert(pos[0]),
                y: LabellingToolSelf.$scale_y.invert(pos[1])};
    };


    LabellingToolSelf._init_key_handlers = function() {
        __labelling_tool_key_handler.handler = LabellingToolSelf._on_key_down;
    };

    LabellingToolSelf._shutdown_key_handlers = function() {
        __labelling_tool_key_handler.handler = null;
    };

    LabellingToolSelf._on_key_down = function(event) {
        if (LabellingToolSelf.$tool !== null) {
            LabellingToolSelf.$tool.on_key_down(event);
        }
    };


    return LabellingToolSelf;
}


var __labelling_tool_key_handler = {};

__labelling_tool_key_handler.handler = null;
__labelling_tool_key_handler.connected = false;

