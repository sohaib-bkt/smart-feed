import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { PreferencesProvider } from './src/context/PreferencesContext';
import AppNavigator from './src/navigation/AppNavigator';

export default function App() {
  return (
    <SafeAreaProvider>
      <PreferencesProvider>
        <StatusBar style="auto" />
        <AppNavigator />
      </PreferencesProvider>
    </SafeAreaProvider>
  );
}