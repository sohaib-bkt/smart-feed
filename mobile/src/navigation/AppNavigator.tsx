import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Text } from 'react-native';
import FeedScreen from '../screens/FeedScreen';
import SettingsScreen from '../screens/SettingsScreen';
import SearchScreen from '../screens/SearchScreen';
import ProfileScreen from '../screens/ProfileScreen';
import { Colors } from '../theme/theme';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

function FeedStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen
        name="FeedMain"
        component={FeedScreen}
        options={{ headerShown: false }}
      />
    </Stack.Navigator>
  );
}

export default function AppNavigator() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused }) => {
            const icons: Record<string, string> = {
              Feed: '📱',
              Search: '🔍',
              Profile: '👤',
              Settings: '⚙️',
            };
            return (
              <Text style={{ fontSize: focused ? 24 : 20 }}>
                {icons[route.name] || '📄'}
              </Text>
            );
          },
          tabBarActiveTintColor: Colors.primary,
          tabBarInactiveTintColor: Colors.textLight,
          tabBarStyle: {
            backgroundColor: Colors.surface,
            borderTopColor: Colors.border,
          },
          headerStyle: {
            backgroundColor: Colors.surface,
          },
          headerTitleStyle: {
            color: Colors.text,
            fontWeight: '600',
          },
        })}
      >
        <Tab.Screen
          name="Feed"
          component={FeedStack}
          options={{ headerShown: false, tabBarLabel: 'Feed' }}
        />
        <Tab.Screen
          name="Search"
          component={SearchScreen}
          options={{ title: 'Recherche' }}
        />
        <Tab.Screen
          name="Profile"
          component={ProfileScreen}
          options={{ title: 'Profil' }}
        />
        <Tab.Screen
          name="Settings"
          component={SettingsScreen}
          options={{ title: 'Paramètres' }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}